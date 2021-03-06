import math
import numpy as np

from analysis_engine import settings

from analysis_engine.library import (
    all_of,
    any_of,
    bearing_and_distance,
    closest_unmasked_value,
    cycle_finder,
    cycle_match,
    find_dlcs,
    find_edges,
    first_order_washout,
    first_valid_sample,
    index_at_value,
    index_at_value_or_level_off,
    is_index_within_slices,
    is_index_within_slice,
    is_slice_within_slice,
    last_valid_sample,
    moving_average,
    nearest_neighbour_mask_repair,
    peak_curvature,
    rate_of_change,
    rate_of_change_array,
    repair_mask,
    runs_of_ones,
    shift_slice,
    shift_slices,
    slice_duration,
    slices_and,
    slices_from_to,
    slices_not,
    slices_or,
    slices_overlap,
    slices_remove_small_gaps,
    slices_remove_small_slices,
)

from analysis_engine.node import A, FlightPhaseNode, P, S, KTI, M

from analysis_engine.settings import (
    AIRBORNE_THRESHOLD_TIME,
    AIRSPEED_THRESHOLD,
    BOUNCED_LANDING_THRESHOLD,
    BOUNCED_MAXIMUM_DURATION,
    DESCENT_LOW_CLIMB_THRESHOLD,
    GROUNDSPEED_FOR_MOBILE,
    HEADING_RATE_FOR_MOBILE,
    HEADING_TURN_OFF_RUNWAY,
    HEADING_TURN_ONTO_RUNWAY,
    HOLDING_MAX_GSPD,
    HOLDING_MIN_TIME,
    HYSTERESIS_FPALT_CCD,
    INITIAL_CLIMB_THRESHOLD,
    INITIAL_APPROACH_THRESHOLD,
    KTS_TO_MPS,
    LANDING_THRESHOLD_HEIGHT,
    RATE_OF_TURN_FOR_FLIGHT_PHASES,
    RATE_OF_TURN_FOR_TAXI_TURNS,
    REJECTED_TAKEOFF_THRESHOLD,
    TAKEOFF_ACCELERATION_THRESHOLD,
    VERTICAL_SPEED_FOR_CLIMB_PHASE,
    VERTICAL_SPEED_FOR_DESCENT_PHASE,
)


class Airborne(FlightPhaseNode):
    '''
    Periods where the aircraft is in the air, includes periods where on the
    ground for short periods (touch and go).
    
    TODO: Review whether depending upon the "dips" calculated by Altitude AAL
    would be more sensible as this will allow for negative AAL values longer
    than the remove_small_gaps time_limit.
    '''
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast')):
        
        # Remove short gaps in going fast to account for aerobatic manoeuvres 
        speedy_slices = slices_remove_small_gaps(fast.get_slices(),
                                                 time_limit=60, hz=fast.frequency)
        
        # Just find out when altitude above airfield is non-zero.
        for speedy in speedy_slices:
            # Stop here if the aircraft never went fast.
            if speedy.start is None and speedy.stop is None:
                break

            start_point = speedy.start or 0
            stop_point = speedy.stop or len(alt_aal.array)
            # Restrict data to the fast section (it's already been repaired)
            working_alt = alt_aal.array[start_point:stop_point]

            # Stop here if there is inadequate airborne data to process.
            if working_alt is None:
                break
            airs = slices_remove_small_gaps(
                np.ma.clump_unmasked(np.ma.masked_less_equal(working_alt, 0.0)),
                time_limit=40, # 10 seconds was too short for Herc which flies below 0  AAL for 30 secs.
                hz=alt_aal.frequency)
            # Make sure we propogate None ends to data which starts or ends in
            # midflight.
            for air in airs:
                begin = air.start
                if begin + start_point == 0: # Was in the air at start of data
                    begin = None
                end = air.stop
                if end + start_point >= len(alt_aal.array): # Was in the air at end of data
                    end = None
                if begin is None or end is None:
                    self.create_phase(shift_slice(slice(begin, end),
                                                  start_point))
                else:
                    duration = end - begin
                    if (duration / alt_aal.hz) > AIRBORNE_THRESHOLD_TIME:
                        self.create_phase(shift_slice(slice(begin, end),
                                                      start_point))


class GoAroundAndClimbout(FlightPhaseNode):
    '''
    We already know that the Key Time Instance has been identified at the
    lowest point of the go-around, and that it lies below the 3000ft
    approach thresholds. The function here is to expand the phase 500ft before
    to the first level off after (up to 2000ft maximum).
    '''

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               gas=KTI('Go Around')):
        # Find the ups and downs in the height trace.
        alt_idxs, alt_vals = find_dlcs(alt_aal.array)
        # Smooth over very small negative rates of change in altitude to
        # avoid index at closest value returning the slight negative change
        # in place of the real altitude peak where the 500ft or 2000ft
        # thresholds are not reached.
        
        # quite a bit of smoothing is required to remove bumpy altitude signals
        smoothed_alt = moving_average(alt_aal.array, window=15)
        for ga in gas:
            ga_idx = ga.index
            #---------- Scan to match the altitude minima ----------
            # A wide tolerance is required to cater for lazy go arounds over rough ground
            prev_idx, post_idx = cycle_match(ga_idx, alt_idxs, dist=1000)
            #--------------- Go-Around Altitude ---------------
            # Find the go-around altitude
            index, value = closest_unmasked_value(
                alt_aal.array, ga_idx, 
                slice(prev_idx,  # previous peak index
                      post_idx)  # next peak index
            )
            #--------------- 500ft before ---------------
            # We have matched the cycle to the (possibly radio height
            # based) go-around KTI.
            # Establish an altitude range around this point
            start_slice = slice(index, prev_idx, -1)  # work backwards towards previous peak
            ga_start = index_at_value_or_level_off(smoothed_alt, 
                                                   value+500, start_slice)
            #--------------- Level off or 2000ft after ---------------
            stop_slice = slice(index, post_idx)  # look forwards towards next peak
            # find the nearest value; we are protected by the cycle peak
            # as the slice.stop from going too far forward.
            ga_stop = index_at_value_or_level_off(smoothed_alt,
                                                  value+2000, stop_slice)
            # round to nearest positions, with a simple reasonableness check.
            if math.ceil(ga_stop) > int(ga_start):
                self.create_phase(slice(int(ga_start), math.ceil(ga_stop)))
        #endfor each goaround
        return


class Holding(FlightPhaseNode):
    """
    Holding is a process which involves multiple turns in a short period,
    normally in the same sense. We therefore compute the average rate of turn
    over a long period to reject short turns and pass the entire holding
    period.

    Note that this is the only function that should use "Heading Increasing"
    as we are only looking for turns, and not bothered about the sense or
    actual heading angle.
    """
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               hdg=P('Heading Increasing'),
               lat=P('Latitude Smoothed'), lon=P('Longitude Smoothed')):
        _, height_bands = slices_from_to(alt_aal.array, 20000, 3000)
        # Three minutes should include two turn segments.
        turn_rate = rate_of_change(hdg, 3 * 60)
        for height_band in height_bands:
            # We know turn rate will be positive because Heading Increasing only
            # increases.
            turn_bands = np.ma.clump_unmasked(
                np.ma.masked_less(turn_rate[height_band], 0.5))
            hold_bands=[]
            for turn_band in shift_slices(turn_bands, height_band.start):
                # Reject short periods and check that the average groundspeed was
                # low. The index is reduced by one sample to avoid overruns, and
                # this is fine because we are not looking for great precision in
                # this test.
                hold_sec = turn_band.stop - turn_band.start
                if (hold_sec > HOLDING_MIN_TIME*alt_aal.frequency):
                    start = turn_band.start
                    stop = turn_band.stop - 1
                    _, hold_dist = bearing_and_distance(
                        lat.array[start], lon.array[start],
                        lat.array[stop], lon.array[stop])
                    if hold_dist/KTS_TO_MPS/hold_sec < HOLDING_MAX_GSPD:
                        hold_bands.append(turn_band)

            self.create_phases(hold_bands)


class EngHotelMode(FlightPhaseNode):
    '''
    Some turbo props use the Engine 2 turbine to provide power and air whilst
    the aircraft is on the ground, a brake is applied to prevent the
    propellers from rotating
    '''

    @classmethod
    def can_operate(cls, available, family=A('Family')):
        return all_of(('Eng (2) Np', 'Eng (1) N1', 'Eng (2) N1', 'Grounded'), available) \
            and family.value in ('ATR-42', 'ATR-72') # Not all aircraft with Np will have a 'Hotel' mode
        

    def derive(self, eng2_np=P('Eng (2) Np'),
               eng1_n1=P('Eng (1) N1'), eng2_n1=P('Eng (2) N1'), groundeds=S('Grounded')):
        pos_hotel = (eng2_n1.array > 45) & (eng2_np.array <= 0) & (eng1_n1.array < 40)
        hotel_mode = slices_and(runs_of_ones(pos_hotel), groundeds.get_slices())
        self.create_phases(hotel_mode)



class ApproachAndLanding(FlightPhaseNode):
    '''
    Approaches from 3000ft to lowest point in the approach (where a go around
    is performed) or down to and including the landing phase.
    
    Q: Suitable to replace this with BottomOfDescent and working back from
    those KTIs rather than having to deal with GoAround AND Landings?
    '''
    # Force to remove problem with desynchronising of approaches and landings
    # (when offset > 0.5)
    align_offset = 0

    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'),
               lands=S('Landing'), gas=KTI('Go Around')):
        # Prepare to extract the slices
        app_slices = []
        ga_slices = []

        # Find the ups and downs in the height trace to restrict search ranges
        ##cycle_idxs, _ = cycle_finder(alt_aal.array, min_step=500.0)
        cycle_idxs, _ = find_dlcs(alt_aal.array)
        for land in lands:
            prev_peak, _ = cycle_match(land.slice.start, cycle_idxs, dist=10000)
            _slice = slice(land.slice.start, prev_peak, -1)
            app_start = index_at_value_or_level_off(
                alt_aal.array, INITIAL_APPROACH_THRESHOLD, _slice)
            app_slices.append(slice(app_start, land.slice.stop))

        for ga in gas:
            # Establish the altitude up to 3000ft before go-around. We know
            # we are below 3000ft as that's the definition of the Go-Around
            # (below 3000ft followed by climb of 500ft).
            ga_start = index_at_value(alt_aal.array, 3000, 
                                      slice(ga.index, None, -1),
                                      endpoint='closing')
            ga_slices.append(slice(ga_start, ga.index+1))

        all_apps = slices_or(app_slices, ga_slices)
        if not all_apps:
            self.warning('Flight with no valid approach or go-around phase. '
                         'Probably truncated data')
        else:
            self.create_phases(all_apps)


class Approach(FlightPhaseNode):
    """
    This separates out the approach phase excluding the landing.
    
    Includes all approaches such as Go Arounds, but does not include any
    climbout afterwards.
    
    Landing starts at 50ft, therefore this phase is until 50ft.
    """
    def derive(self, apps=S('Approach And Landing'), lands=S('Landing')):
        app_slices = []
        begin = None
        end = None
        land_slices = []
        for app in apps:
            _slice = app.slice
            app_slices.append(_slice)
            if begin is None:
                begin = _slice.start
                end = _slice.stop
            else:
                begin = min(begin, _slice.start)
                end = max(end, _slice.stop)
        for land in lands:
            land_slices.append(land.slice)
        self.create_phases(slices_and(app_slices,
                                      slices_not(land_slices,
                                                 begin_at=begin,
                                                 end_at=end)))


class BouncedLanding(FlightPhaseNode):
    '''
    TODO: Review increasing the frequency for more accurate indexing into the
    altitude arrays.

    Q: Should Airborne be first so we align to its offset?
    '''
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases'), 
               airs=S('Airborne'),
               fast=S('Fast')):
        for speedy in fast:
            for air in airs:
                if slices_overlap(speedy.slice, air.slice):
                    start = air.slice.stop
                    stop = speedy.slice.stop
                    if (stop - start) / self.frequency > BOUNCED_MAXIMUM_DURATION:
                        # duration too long to be a bounced landing!
                        # possible cause: Touch and go.
                        continue
                    elif start == stop:
                        stop += 1
                    scan = alt_aal.array[start:stop]
                    ht = max(scan)
                    if ht > BOUNCED_LANDING_THRESHOLD:
                        #TODO: Input maximum BOUNCE_HEIGHT check?
                        up = np.ma.clump_unmasked(np.ma.masked_less_equal(scan,
                                                                          0.0))
                        self.create_phase(
                            shift_slice(slice(up[0].start, up[-1].stop), start))


class ClimbCruiseDescent(FlightPhaseNode):
    def derive(self, alt_std=P('Altitude STD Smoothed'),
               airs=S('Airborne')):
        for air in airs:
            altitudes = alt_std.array[air.slice]
            # We squash the altitude signal above 10,000ft so that changes of
            # altitude to create a new flight phase have to be 10 times
            # greater; 500ft changes below 10,000ft are significant, while
            # above this 5,000ft is more meaningful.
            alt_squash = np.ma.where(altitudes>10000,
                                     (altitudes-10000)/10.0+10000,
                                     altitudes
                                     )
            pk_idxs, pk_vals = cycle_finder(alt_squash,
                                            min_step=HYSTERESIS_FPALT_CCD)
            
            if pk_vals is not None:
                n = 0
                pk_idxs += air.slice.start or 0
                n_vals = len(pk_vals)
                while n < n_vals - 1:
                    pk_val = pk_vals[n]
                    pk_idx = pk_idxs[n]
                    next_pk_val = pk_vals[n + 1]
                    next_pk_idx = pk_idxs[n + 1]
                    if pk_val > next_pk_val:
                        # descending
                        self.create_phase(slice(None, next_pk_idx))
                        n += 1
                    else:
                        # ascending
                        # We are going upwards from n->n+1, does it go down
                        # again?
                        if n + 2 < n_vals:
                            if pk_vals[n + 2] < next_pk_val:
                                # Hurrah! make that phase
                                self.create_phase(slice(pk_idx,
                                                        pk_idxs[n + 2]))
                                n += 2
                        else:
                            self.create_phase(slice(pk_idx, None))
                            n += 1


"""
class CombinedClimb(FlightPhaseNode):
    '''
    Climb phase from liftoff or go around to top of climb
    '''
    def derive(self,
               toc=KTI('Top Of Climb'),
               ga=KTI('Go Around'),
               lo=KTI('Liftoff'),
               touchdown=KTI('Touchdown')):

        end_list = [x.index for x in toc.get_ordered_by_index()]
        start_list = [y.index for y in [lo.get_first()] + ga.get_ordered_by_index()]
        assert len(start_list) == len(end_list)

        slice_idxs = zip(start_list, end_list)
        for slice_tuple in slice_idxs:
            self.create_phase(slice(*slice_tuple))
"""

class Climb(FlightPhaseNode):
    '''
    This phase goes from 1000 feet (top of Initial Climb) in the climb to the
    top of climb
    '''
    def derive(self,
               toc=KTI('Top Of Climb'),
               eot=KTI('Climb Start'), # AKA End Of Initial Climb
               bod=KTI('Bottom Of Descent')):
        # First we extract the kti index values into simple lists.
        toc_list = []
        for this_toc in toc:
            toc_list.append(this_toc.index)

        # Now see which follows a takeoff
        for this_eot in eot:
            eot = this_eot.index
            # Scan the TOCs
            closest_toc = None
            for this_toc in toc_list:
                if (eot < this_toc and
                    (this_toc < closest_toc
                     or
                     closest_toc is None)):
                    closest_toc = this_toc
            # Build the slice from what we have found.
            self.create_phase(slice(eot, closest_toc))


class Climbing(FlightPhaseNode):
    def derive(self, vert_spd=P('Vertical Speed For Flight Phases'),
               airs=S('Airborne')):
        # Climbing is used for data validity checks and to reinforce regimes.
        for air in airs:
            climbing = np.ma.masked_less(vert_spd.array[air.slice],
                                         VERTICAL_SPEED_FOR_CLIMB_PHASE)
            climbing_slices = slices_remove_small_gaps(
                np.ma.clump_unmasked(climbing), time_limit=30.0, hz=vert_spd.hz)
            self.create_phases(shift_slices(climbing_slices, air.slice.start))


class Cruise(FlightPhaseNode):
    def derive(self,
               ccds=S('Climb Cruise Descent'),
               tocs=KTI('Top Of Climb'),
               tods=KTI('Top Of Descent')):
        # We may have many phases, tops of climb and tops of descent at this
        # time.
        # The problem is that they need not be in tidy order as the lists may
        # not be of equal lengths.
        for ccd in ccds:
            toc = tocs.get_first(within_slice=ccd.slice)
            if toc:
                begin = toc.index
            else:
                begin = ccd.slice.start

            tod = tods.get_last(within_slice=ccd.slice)
            if tod:
                end = tod.index
            else:
                end = ccd.slice.stop

            # Some flights just don't cruise. This can cause headaches later
            # on, so we always cruise for at least one second !
            if end <= begin:
                end = begin + 1

            self.create_phase(slice(begin,end))


class InitialCruise(FlightPhaseNode):
    '''
    This is a period from five minutes into the cruise lasting for 30
    seconds, and is used to establish average conditions for fuel monitoring
    programmes.
    '''
    
    align_frequency = 1.0
    align_offset = 0.0
    
    def derive(self, cruises=S('Cruise')):
        cruise = cruises[0].slice
        if cruise.stop - cruise.start > 330:
            self.create_phase(slice(cruise.start+300, cruise.start+330))
            

class CombinedDescent(FlightPhaseNode):
    def derive(self,
               tod_set=KTI('Top Of Descent'),
               bod_set=KTI('Bottom Of Descent'),
               liftoff=KTI('Liftoff'),
               touchdown=KTI('Touchdown')):

        end_list = [x.index for x in bod_set.get_ordered_by_index()]
        start_list = [y.index for y in tod_set.get_ordered_by_index()]
        assert len(start_list) == len(end_list)

        slice_idxs = zip(start_list, end_list)
        for slice_tuple in slice_idxs:
            self.create_phase(slice(*slice_tuple))


class Descending(FlightPhaseNode):
    """
    Descending faster than 500fpm towards the ground
    """
    def derive(self, vert_spd=P('Vertical Speed For Flight Phases'),
               airs=S('Airborne')):
        # Vertical speed limits of 500fpm gives good distinction with level
        # flight.
        for air in airs:
            descending = np.ma.masked_greater(vert_spd.array[air.slice],
                                              VERTICAL_SPEED_FOR_DESCENT_PHASE)
            desc_slices = slices_remove_small_slices(np.ma.clump_unmasked(descending))
            self.create_phases(shift_slices(desc_slices, air.slice.start))


class Descent(FlightPhaseNode):
    def derive(self,
               tod_set=KTI('Top Of Descent'),
               bod_set=KTI('Bottom Of Descent')):
        # First we extract the kti index values into simple lists.
        tod_list = []
        for this_tod in tod_set:
            tod_list.append(this_tod.index)

        # Now see which preceded this minimum
        for this_bod in bod_set:
            bod = this_bod.index
            # Scan the TODs
            closest_tod = None
            for this_tod in tod_list:
                if (bod > this_tod and
                    this_tod > closest_tod):
                    closest_tod = this_tod

            # Build the slice from what we have found.
            self.create_phase(slice(closest_tod, bod))
        return


class DescentToFlare(FlightPhaseNode):
    '''
    Descent phase down to 50ft.
    '''

    def derive(self,
            descents=S('Descent'),
            alt_aal=P('Altitude AAL For Flight Phases')):
        #TODO: Ensure we're still in the air
        for descent in descents:
            end = index_at_value(alt_aal.array, 50.0, descent.slice)
            if end is None:
                end = descent.slice.stop
            self.create_phase(slice(descent.slice.start, end))


class DescentLowClimb(FlightPhaseNode):
    '''
    Finds where the aircaft descends below the INITIAL_APPROACH_THRESHOLD and
    then climbs out again - an indication of a go-around.
    
    TODO: Consider refactoring this based on the Bottom Of Descent KTIs and
    just check the altitude at each BOD.
    '''
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases')):
        pk_idxs, pk_vals = find_dlcs(alt_aal.array)
            
        if pk_vals is None or len(pk_vals) < 3:
            return 
        
        for n in range(1, len(pk_vals) - 1):
            if (pk_vals[n-1]-pk_vals[n]) > DESCENT_LOW_CLIMB_THRESHOLD and \
               (pk_vals[n+1]-pk_vals[n]) > DESCENT_LOW_CLIMB_THRESHOLD:
                self.create_phase(slice(pk_idxs[n-1], pk_idxs[n+1]))


class Fast(FlightPhaseNode):
    '''
    Data will have been sliced into single flights before entering the
    analysis engine, so we can be sure that there will be only one fast
    phase. This may have masked data within the phase, but by taking the
    notmasked edges we enclose all the data worth analysing.

    Therefore len(Fast) in [0,1]

    TODO: Discuss whether this assertion is reliable in the presence of air data corruption.
    '''

    def derive(self, airspeed=P('Airspeed For Flight Phases')):
        """
        Did the aircraft go fast enough to possibly become airborne?

        # We use the same technique as in index_at_value where transition of
        # the required threshold is detected by summing shifted difference
        # arrays. This has the particular advantage that we can reject
        # excessive rates of change related to data dropouts which may still
        # pass the data validation stage.
        value_passing_array = (airspeed.array[0:-2]-AIRSPEED_THRESHOLD) * \
            (airspeed.array[1:-1]-AIRSPEED_THRESHOLD)
        test_array = np.ma.masked_outside(value_passing_array, 0.0, -100.0)
        """
        fast_samples = np.ma.clump_unmasked(
            np.ma.masked_less(airspeed.array, AIRSPEED_THRESHOLD))

        for fast_sample in fast_samples:
            start = fast_sample.start
            stop = fast_sample.stop
            if abs(airspeed.array[start] - AIRSPEED_THRESHOLD) > 20:
                start = None
            if abs(airspeed.array[stop - 1] - AIRSPEED_THRESHOLD) > 30:
                stop = None
            # Dont create a phase if neither is valid.
            if start or stop:
                self.create_phase(slice(start, stop))


class FinalApproach(FlightPhaseNode):
    def derive(self, alt_aal=P('Altitude AAL For Flight Phases')):
        self.create_phases(alt_aal.slices_from_to(1000, 50))


class GearExtending(FlightPhaseNode):
    '''
    Gear extending and retracting are section nodes, as they last for a
    finite period. Based on the Gear Red Warnings.

    For some aircraft no parameters to identify the transit are recorded, so
    a nominal period is included in Gear Down Selected Calculations to
    allow for exceedance of gear transit limits.
    '''

    def derive(self, gear_down_selected=M('Gear Down Selected'),
               gear_down=M('Gear Down'), airs=S('Airborne')):
        
        in_transit = (gear_down_selected.array == 'Down') & (gear_down.array != 'Down')
        gear_extending = slices_and(runs_of_ones(in_transit), airs.get_slices())
        self.create_phases(gear_extending)


class GearExtended(FlightPhaseNode):
    '''
    Simple phase translation of the Gear Down parameter.
    '''
    def derive(self, gear_down=M('Gear Down')):
        repaired = repair_mask(gear_down.array, gear_down.frequency, 
                               repair_duration=120, extrapolate=True)
        self.create_phases(runs_of_ones(repaired == 'Down'))


class GearRetracting(FlightPhaseNode):
    '''
    Gear extending and retracting are section nodes, as they last for a
    finite period. Based on the Gear Red Warnings.

    For some aircraft no parameters to identify the transit are recorded, so
    a nominal period is included in Gear Up Selected Calculations to
    allow for exceedance of gear transit limits.
    '''

    def derive(self, gear_up_selected=M('Gear Up Selected'),
               gear_down=M('Gear Down'), airs=S('Airborne')):
        
        in_transit = (gear_up_selected.array == 'Up') & (gear_down.array != 'Up')
        gear_retracting = slices_and(runs_of_ones(in_transit), airs.get_slices())
        self.create_phases(gear_retracting)


class GearRetracted(FlightPhaseNode):
    '''
    Simple phase translation of the Gear Down parameter to show gear Up.
    '''
    def derive(self, gear_down=M('Gear Down')):
        #TODO: self = 1 - 'Gear Extended'
        repaired = repair_mask(gear_down.array, gear_down.frequency, 
                               repair_duration=120, extrapolate=True)
        self.create_phases(runs_of_ones(repaired == 'Up'))


def scan_ils(beam, ils_dots, height, scan_slice, frequency, duration=10):
    '''
    Scans ils dots and returns last slice where ils dots fall below 1 and remain below 2.5 dots
    if beam is glideslope slice will not extend below 200ft.

    :param beam: 'localizer' or 'glideslope'
    :type beam: str
    :param ils_dots: 'localizer' or 'glideslope'
    :type ils_dots: str
    :param height: 'localizer' or 'glideslope'
    :type height: str
    :param scan_slice: 'localizer' or 'glideslope'
    :type scan_slice: str
    :param frequency: input signal sample rate
    :type frequency: float
    :param duration: Minimum duration for the ILS to be established
    :type duration: float, default = 10 seconds.
    '''
    if beam not in ['localizer', 'glideslope']:
        raise ValueError('Unrecognised beam type in scan_ils')

    if np.ma.count(ils_dots[scan_slice]) < duration*frequency:
        # less than duration seconds of valid data within slice
        return None

    # Find the range of valid ils dots withing scan slice
    valid_ends = np.ma.flatnotmasked_edges(ils_dots[scan_slice])
    if valid_ends is None:
        return None
    valid_slice = slice(*(valid_ends+scan_slice.start))
    if np.ma.count(ils_dots[valid_slice])/float(len(ils_dots[valid_slice])) < 0.4:
        # less than 40% valid data within valid data slice
        return None

    # get abs of ils dots as its used everywhere and repair small masked periods
    ils_abs = repair_mask(np.ma.abs(ils_dots), frequency=frequency, repair_duration=5)

    # ----------- Find loss of capture

    last_valid_idx, last_valid_value = last_valid_sample(ils_abs[scan_slice])

    if last_valid_value < 2.5:
        # finished established ? if established in first place
        ils_lost_idx = scan_slice.start + last_valid_idx + 1
    else:
        # find last time went below 2.5 dots
        last_25_idx = index_at_value(ils_abs, 2.5, slice(scan_slice.stop, scan_slice.start, -1))
        if last_25_idx is None:
            # never went below 2.5 dots
            return None
        else:
            ils_lost_idx = last_25_idx

    if beam == 'glideslope':
        # If Glideslope find index of height last passing 200ft and use the
        # smaller of that and any index where the ILS was lost
        idx_200 = index_at_value(height, 200, slice(scan_slice.stop,
                                                scan_slice.start, -1),
                             endpoint='closing')
        if idx_200 is not None:
            ils_lost_idx = min(ils_lost_idx, idx_200)

        if np.ma.count(ils_dots[scan_slice.start:ils_lost_idx]) < 5:
            # less than 5 valid values within remaining section
            return None

    # ----------- Find start of capture

    # Find where to start scanning for the point of "Capture", Look for the
    # last time we were within 2.5dots
    scan_start_idx = index_at_value(ils_abs, 2.5, slice(ils_lost_idx-1, scan_slice.start-1, -1))

    if scan_start_idx:
        # Found a point to start scanning from, now look for the ILS goes
        # below 1 dot.
        ils_capture_idx = index_at_value(ils_abs, 1.0, slice(scan_start_idx, ils_lost_idx))
    else:
        # Reached start of section without passing 2.5 dots so check if we
        # started established
        first_valid_idx, first_valid_value = first_valid_sample(ils_abs[slice(scan_slice.start, ils_lost_idx)])

        if first_valid_value < 1.0:
            # started established
            ils_capture_idx = scan_slice.start + first_valid_idx
        else:
            # Find first index of 1.0 dots from start of scan slice
            ils_capture_idx = index_at_value(ils_abs, 1.0, slice(scan_slice.start, ils_lost_idx))

    if ils_capture_idx is None or ils_lost_idx is None:
        return None
    else:
        # OK, we have seen an ILS signal, but let's make sure we did respond
        # to it. The test here is to make sure that we didn't just pass
        # through the beam (L>R or R>L or U>D or D>U) without making an
        # effort to correct the variation.
        ils_slice = slice(ils_capture_idx, ils_lost_idx)
        width = 5.0
        if frequency < 0.5:
            width = 10.0
        ils_rate = rate_of_change_array(ils_dots[ils_slice], frequency, width=width, method='regression')
        top = max(ils_rate)
        bottom = min(ils_rate)
        if top*bottom > 0.0:
            # the signal never changed direction, so went straight through
            # the beam without getting established...
            return None
        else:
            # Hurrah! We did capture the beam
            return ils_slice


class ILSLocalizerEstablished(FlightPhaseNode):
    name = 'ILS Localizer Established'

    @classmethod
    def can_operate(cls, available):
        return all_of(('ILS Localizer',
                       'Altitude AAL For Flight Phases',
                       'Approach And Landing'), available)

    def derive(self, ils_loc=P('ILS Localizer'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               apps=S('Approach And Landing'),
               ils_freq=P('ILS Frequency'),):
        
        def create_ils_phases(slices):
            for _slice in slices:
                ils_slice = scan_ils('localizer', ils_loc.array, alt_aal.array,
                                   _slice, ils_loc.frequency)
                if ils_slice is not None:
                    self.create_phase(ils_slice)
        
        if not ils_freq:
            # If we don't have a frequency source, just scan the signal and
            # hope it was for this runway!
            create_ils_phases(apps.get_slices())
            self.info("No ILS Frequency used. Created %d established phases" % len(self))
            return
        if np.ma.count(ils_freq.array) < 10:
            # ILS frequency tells us that no localizer was established
            self.info("ILS Frequency has no valid data. No established phases created")
            return
        '''
        Note: You can be tuned onto multiple frequencies or the same
              frequency multiple times during each approach.
        '''
        # If we have ILS frequency tuned in check for multiple frequencies
        # using np.ma.around as 110.7 == 110.7 is not always the case when
        # dealing with floats
        frequency_slices = []
        for app_slice in apps.get_slices():
            if not np.ma.count(ils_freq.array[app_slice]):
                # No valid frequency data at all in this slice.
                continue
            # Repair data (without interpolating) during each approach.
            # nn_repair will extrapolate each signal to the start and end of
            # the approach, and we'll fill in the gaps for up to 8 samples
            # from each end (16 repairs in total). Gaps bigger than this will
            # not count towards being established on the ILS.
            ils_freq_repaired = nearest_neighbour_mask_repair(
                ils_freq.array[app_slice], repair_gap_size=16)
            # Look for the changes or when it was not tuned
            frequency_changes = np.ma.diff(np.ma.around(ils_freq_repaired, decimals=2))
            # Create slices for each ILS frequency so they are scanned separately
            app_freq_slices = shift_slices(runs_of_ones(frequency_changes == 0), app_slice.start)
            frequency_slices.extend(app_freq_slices)

        # If we have a frequency source, only create slices if we have some valid frequencies.
        create_ils_phases(frequency_slices)
        self.info("ILS Frequency has valid data. Created %d established phases" % len(self))        


'''
class ILSApproach(FlightPhaseNode):
    name = "ILS Approach"
    """
    Where a Localizer Established phase exists, extend the start and end of
    the phase back to 3 dots (i.e. to beyond the view of the pilot which is
    2.5 dots) and assign this to ILS Approach phase. This period will be used
    to determine the range for the ILS display on the web site and for
    examination for ILS KPVs.
    """
    def derive(self, ils_loc = P('ILS Localizer'),
               ils_loc_ests = S('ILS Localizer Established')):
        # For most of the flight, the ILS will not be valid, so we scan only
        # the periods with valid data, ignoring short breaks:
        locs = np.ma.clump_unmasked(repair_mask(ils_loc.array))
        for loc_slice in locs:
            for ils_loc_est in ils_loc_ests:
                est_slice = ils_loc_est.slice
                if slices_overlap(loc_slice, est_slice):
                    before_established = slice(est_slice.start, loc_slice.start, -1)
                    begin = index_at_value(np.ma.abs(ils_loc.array),
                                                     3.0,
                                                     _slice=before_established)
                    end = est_slice.stop
                    self.create_phase(slice(begin, end))
'''


class ILSGlideslopeEstablished(FlightPhaseNode):
    name = "ILS Glideslope Established"
    """
    Within the Localizer Established phase, compute duration of approach with
    (repaired) Glideslope deviation continuously less than 1 dot,. Where > 10
    seconds, identify as Glideslope Established.
    """
    def derive(self, ils_gs = P('ILS Glideslope'),
               ils_loc_ests = S('ILS Localizer Established'),
               alt_aal=P('Altitude AAL For Flight Phases')):
        # We don't accept glideslope approaches without localizer established
        # first, so this only works within that context. If you want to
        # follow a glidepath without a localizer, seek flight safety guidance
        # elsewhere.
        for ils_loc_est in ils_loc_ests:
            # Only look for glideslope established if the localizer was
            # established.
            if ils_loc_est.slice.start and ils_loc_est.slice.stop:
                gs_est = scan_ils('glideslope', ils_gs.array, alt_aal.array,
                                  ils_loc_est.slice, ils_gs.frequency)
                # If the glideslope signal is corrupt or there is no
                # glidepath (not fitted or out of service) there may be no
                # glideslope established phase, or the proportion of unmasked
                # values may be small.
                if gs_est:
                    good_data = np.ma.count(ils_gs.array[gs_est])
                    all_data = len(ils_gs.array[gs_est]) or 1
                    if (float(good_data)/all_data) < 0.7:
                        self.warning('ILS glideslope signal poor quality in '
                                     'approach - considered not established.')
                        continue
                    self.create_phase(gs_est)


class InitialApproach(FlightPhaseNode):
    def derive(self, alt_AAL=P('Altitude AAL For Flight Phases'),
               app_lands=S('Approach')):
        for app_land in app_lands:
            # We already know this section is below the start of the initial
            # approach phase so we only need to stop at the transition to the
            # final approach phase.
            ini_app = np.ma.masked_where(alt_AAL.array[app_land.slice]<1000,
                                         alt_AAL.array[app_land.slice])
            phases = np.ma.clump_unmasked(ini_app)
            for phase in phases:
                begin = phase.start
                pit = np.ma.argmin(ini_app[phase]) + begin
                if ini_app[pit] < ini_app[begin] :
                    self.create_phases(shift_slices([slice(begin, pit)],
                                                    app_land.slice.start))


class InitialClimb(FlightPhaseNode):
    '''
    Phase from end of Takeoff (35ft) to start of climb (1000ft)
    '''
    def derive(self,
               takeoffs=S('Takeoff'),
               climb_starts=KTI('Climb Start')):
        for takeoff in takeoffs:
            begin = takeoff.stop_edge
            for climb_start in climb_starts.get_ordered_by_index():
                end = climb_start.index
                if end > begin:
                    self.create_phase(slice(begin, end), begin=begin, end=end)
                    break


class LevelFlight(FlightPhaseNode):
    '''
    '''
    def derive(self,
               airs=S('Airborne'),
               vrt_spd=P('Vertical Speed For Flight Phases')):

        for air in airs:
            limit = settings.VERTICAL_SPEED_FOR_LEVEL_FLIGHT
            level_flight = np.ma.masked_outside(vrt_spd.array[air.slice], -limit, limit)
            level_slices = np.ma.clump_unmasked(level_flight)
            level_slices = slices_remove_small_slices(level_slices, 
                                                      time_limit=settings.LEVEL_FLIGHT_MIN_DURATION,
                                                      hz=vrt_spd.frequency)
            self.create_phases(shift_slices(level_slices, air.slice.start))



class StraightAndLevel(FlightPhaseNode):
    '''
    Building on Level Flight, this checks for straight flight. We use heading
    rate as more sensitive than roll attitude and sticking to the core three
    parameters.
    '''
    def derive(self,
               levels=S('Level Flight'),
               hdg=P('Heading')):

        for level in levels:
            limit = settings.HEADING_RATE_FOR_STRAIGHT_FLIGHT
            rot = rate_of_change_array(hdg.array[level.slice], hdg.frequency, width=30)
            straight_flight = np.ma.masked_outside(rot, -limit, limit)
            straight_slices = np.ma.clump_unmasked(straight_flight)
            straight_and_level_slices = slices_remove_small_slices(straight_slices, 
                                                                   time_limit=settings.LEVEL_FLIGHT_MIN_DURATION,
                                                                   hz=hdg.frequency)
            self.create_phases(shift_slices(straight_and_level_slices, level.slice.start))




class Grounded(FlightPhaseNode):
    '''
    Includes start of takeoff run and part of landing run.
    Was "On Ground" but this name conflicts with a recorded 737-6 parameter name.
    '''
    def derive(self, air=S('Airborne'), speed=P('Airspeed For Flight Phases')):
        data_end=len(speed.array)
        gnd_phases = slices_not(air.get_slices(), begin_at=0, end_at=data_end)
        if not gnd_phases:
            # Either all on ground or all in flight.
            median_speed = np.ma.median(speed.array)
            if median_speed > AIRSPEED_THRESHOLD:
                gnd_phases = [slice(None,None,None)]
            else:
                gnd_phases = [slice(0,data_end,None)]

        self.create_phases(gnd_phases)


class Mobile(FlightPhaseNode):
    '''
    This finds the first and last signs of movement to provide endpoints to
    the taxi phases. As Rate Of Turn is derived directly from heading, this
    phase is guaranteed to be operable for very basic aircraft.
    '''
    @classmethod
    def can_operate(cls, available):
        return 'Rate Of Turn' in available

    def derive(self, rot=P('Rate Of Turn'), gspd=P('Groundspeed'),
               toffs=S('Takeoff'), lands=S('Landing')):
        move = np.ma.flatnotmasked_edges(np.ma.masked_less\
                                         (np.ma.abs(rot.array),
                                          HEADING_RATE_FOR_MOBILE))

        if move is None:
            return # for the case where nothing happened

        if gspd:
            # We need to be outside the range where groundspeeds are detected.
            move_gspd = np.ma.flatnotmasked_edges(np.ma.masked_less\
                                                  (np.ma.abs(gspd.array),
                                                   GROUNDSPEED_FOR_MOBILE))
            # moving is a numpy array so needs to be converted to a list of one
            # slice
            move[0] = min(move[0], move_gspd[0])
            move[1] = max(move[1], move_gspd[1])
        else:
            # Without a recorded groundspeed, fall back to the start of the
            # takeoff run and end of the landing run as limits.
            if toffs:
                move[0] = min(move[0], toffs[0].slice.start)
            if lands:
                move[1] = max(move[1], lands[-1].slice.stop)

        self.create_phase(slice(move[0], move[1]))


class Landing(FlightPhaseNode):
    '''
    This flight phase starts at 50 ft in the approach and ends as the
    aircraft turns off the runway. Subsequent KTIs and KPV computations
    identify the specific moments and values of interest within this phase.

    We use Altitude AAL (not "for Flight Phases") to avoid small errors
    introduced by hysteresis, which is applied to avoid hunting in level
    flight conditions, and thereby make sure the 50ft startpoint is exact.
    '''
    def derive(self, head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases'), fast=S('Fast')):
        phases = []
        for speedy in fast:
            # See takeoff phase for comments on how the algorithm works.

            # AARRGG - How can we check if this is at the end of the data
            # without having to go back and test against the airspeed array?
            # TODO: Improve endpoint checks. DJ
            # Answer: 
            #  duration=A('HDF Duration')
            #  array_len = duration.value * self.frequency
            #  if speedy.slice.stop >= array_len: continue
        
            if (speedy.slice.stop is None or \
                speedy.slice.stop >= len(alt_aal.array)):
                break

            landing_run = speedy.slice.stop
            datum = head.array[landing_run]

            first = landing_run - (300 * alt_aal.frequency)
            landing_begin = index_at_value(alt_aal.array,
                                           LANDING_THRESHOLD_HEIGHT,
                                           slice(first, landing_run))
            if landing_begin is None:
                # we are not able to detect a landing threshold height,
                # therefore invalid section
                continue

            # The turn off the runway must lie within eight minutes of the
            # landing. (We did use 5 mins, but found some landings on long
            # runways where the turnoff did not happen for over 6 minutes
            # after touchdown).
            last = landing_run + (480 * head.frequency)

            # A crude estimate is given by the angle of turn
            landing_end = index_at_value(np.ma.abs(head.array-datum),
                                         HEADING_TURN_OFF_RUNWAY,
                                         slice(landing_run, last))
            if landing_end is None:
                # The data ran out before the aircraft left the runway so use
                # all we have.
                landing_end = len(head.array)-1

            # ensure any overlap with phases are ignored (possibly due to
            # data corruption returning multiple fast segments)
            new_phase = [slice(landing_begin, landing_end)]
            phases = slices_or(phases, new_phase)
        self.create_phases(phases)
        


class LandingRoll(FlightPhaseNode):
    '''
    FDS developed this node to support the UK CAA Significant Seven
    programme. This phase is used when computing KPVs relating to the
    deceleration phase of the landing.

    "CAA to go with T/D to 60 knots with the T/D defined as less than 2 deg
    pitch (after main gear T/D)."

    The complex index_at_value ensures that if the aircraft does not flare to
    2 deg, we still capture the highest attitude as the start of the landing
    roll, and the landing roll starts as the aircraft passes 2 deg the last
    time, i.e. as the nosewheel comes down and not as the flare starts.
    '''
    @classmethod
    def can_operate(cls, available):
        return 'Landing' in available and any_of(('Airspeed True', 'Groundspeed'), available)

    def derive(self, pitch=P('Pitch'), gspd=P('Groundspeed'),
               aspd=P('Airspeed True'), lands=S('Landing')):
        if gspd:
            speed = gspd.array
        else:
            speed = aspd.array
        for land in lands:
            # Airspeed True on some aircraft do not record values below 61
            end = index_at_value(speed, 65.0, land.slice)
            if end is None:
                # due to masked values, use the land.stop rather than
                # searching from the end of the data
                end = land.slice.stop
            begin = None
            if pitch:
                begin = index_at_value(pitch.array, 2.0,
                                       slice(end,land.slice.start,-1),
                                       endpoint='nearest')
            if begin is None:
                # due to masked values, use land.start in place
                begin = land.slice.start

            self.create_phase(slice(begin, end), begin=begin, end=end)


class RejectedTakeoff(FlightPhaseNode):
    '''
    Rejected Takeoff based on Acceleration Longitudinal Offset Removed exceeding
    the TAKEOFF_ACCELERATION_THRESHOLD and not being followed by a liftoff.
    '''
    
    def derive(self, accel_lon=P('Acceleration Longitudinal Offset Removed'),
               groundeds=S('Grounded')):
        accel_lon_smoothed = moving_average(accel_lon.array)
        
        accel_lon_masked = np.ma.copy(accel_lon_smoothed)
        accel_lon_masked.mask |= accel_lon_masked <= TAKEOFF_ACCELERATION_THRESHOLD
        
        accel_lon_slices = np.ma.clump_unmasked(accel_lon_masked)
        
        potential_rtos = []
        for grounded in groundeds:
            for accel_lon_slice in accel_lon_slices:
                if is_index_within_slice(accel_lon_slice.start, grounded.slice) and \
                   is_index_within_slice(accel_lon_slice.stop, grounded.slice):
                    potential_rtos.append(accel_lon_slice)
            
        for next_index, potential_rto in enumerate(potential_rtos, start=1):
            # we get the min of the potential rto stop and the end of the
            # data for cases where the potential rto is detected close to the
            # end of the data
            check_grounded_idx = min(potential_rto.stop + 60 * self.frequency,
                                     len(accel_lon.array) - 1)
            if is_index_within_slices(check_grounded_idx, groundeds.get_slices()):
                # if soon after potential rto and still grounded we have a
                # rto, otherwise we continue to takeoff
                self.create_phase(slice(max(potential_rto.start-(10 * self.hz), 0),
                                    min(potential_rto.stop+(30 * self.hz),
                                        len(accel_lon.array))))


class Takeoff(FlightPhaseNode):
    """
    This flight phase starts as the aircraft turns onto the runway and ends
    as it climbs through 35ft. Subsequent KTIs and KPV computations identify
    the specific moments and values of interest within this phase.

    We use Altitude AAL (not "for Flight Phases") to avoid small errors
    introduced by hysteresis, which is applied to avoid hunting in level
    flight conditions, and make sure the 35ft endpoint is exact.
    """
    def derive(self, head=P('Heading Continuous'),
               alt_aal=P('Altitude AAL For Flight Phases'),
               fast=S('Fast')):

        # Note: This algorithm works across the entire data array, and
        # not just inside the speedy slice, so the final indexes are
        # absolute and not relative references.

        for speedy in fast:
            # This basic flight phase cuts data into fast and slow sections.

            # We know a takeoff should come at the start of the phase,
            # however if the aircraft is already airborne, we can skip the
            # takeoff stuff.
            if speedy.slice.start is None:
                break

            # The aircraft is part way down its takeoff run at the start of
            # the section.
            takeoff_run = speedy.slice.start

            #-------------------------------------------------------------------
            # Find the start of the takeoff phase from the turn onto the runway.

            # The heading at the start of the slice is taken as a datum for now.
            datum = head.array[takeoff_run]

            # Track back to the turn
            # If he took more than 5 minutes on the runway we're not interested!
            first = max(0, takeoff_run - (300 * head.frequency))
            takeoff_begin = index_at_value(np.ma.abs(head.array - datum),
                                           HEADING_TURN_ONTO_RUNWAY,
                                           slice(takeoff_run, first, -1))

            # Where the data starts in line with the runway, default to the
            # start of the data
            if takeoff_begin is None:
                takeoff_begin = first

            #-------------------------------------------------------------------
            # Find the end of the takeoff phase as we climb through 35ft.

            # If it takes more than 5 minutes, he's certainly not doing a normal
            # takeoff !
            last = takeoff_run + (300 * alt_aal.frequency)
            takeoff_end = index_at_value(alt_aal.array, INITIAL_CLIMB_THRESHOLD,
                                         slice(takeoff_run, last))

            if takeoff_end <= 0:
                # catches if None or zero
                continue

            #-------------------------------------------------------------------
            # Create a phase for this takeoff
            self.create_phase(slice(takeoff_begin, takeoff_end))


class TakeoffRoll(FlightPhaseNode):
    '''
    Sub-phase originally written for the correlation tests but has found use
    in the takeoff KPVs where we are interested in the movement down the
    runway, not the turnon or liftoff.

    If pitch is not avaliable to detect rotation we use the end of the takeoff.
    '''

    @classmethod
    def can_operate(cls, available):
        return all_of(('Takeoff', 'Takeoff Acceleration Start'), available)

    def derive(self, toffs=S('Takeoff'),
               acc_starts=KTI('Takeoff Acceleration Start'),
               pitch=P('Pitch')):
        for toff in toffs:
            begin = toff.slice.start # Default if acceleration term not available.
            if acc_starts: # We don't bother with this for data validation, hence the conditional
                acc_start = acc_starts.get_last(within_slice=toff.slice)
                if acc_start:
                    begin = acc_start.index
            chunk = slice(begin, toff.slice.stop)
            if pitch:
                pwo = first_order_washout(pitch.array[chunk], 3.0, pitch.frequency)
                two_deg_idx = index_at_value(pwo, 2.0)
                if two_deg_idx is None:
                    roll_end = toff.slice.stop
                    self.warning('Aircraft did not reach a pitch of 2 deg or Acceleration Start is incorrect')
                else:
                    roll_end = two_deg_idx + begin
                self.create_phase(slice(begin, roll_end))
            else:
                self.create_phase(chunk)


class TakeoffRotation(FlightPhaseNode):
    '''
    This is used by correlation tests to check control movements during the
    rotation and lift phases.
    '''
    def derive(self, lifts=S('Liftoff')):
        if not lifts:
            return
        lift_index = lifts.get_first().index
        start = lift_index - 10
        end = lift_index + 15
        self.create_phase(slice(start, end))


################################################################################
# Takeoff/Go-Around Ratings


# TODO: Write some unit tests!
class Takeoff5MinRating(FlightPhaseNode):
    '''
    For engines, the period of high power operation is normally 5 minutes from
    the start of takeoff. Also applies in the case of a go-around.
    '''
    def derive(self, toffs=S('Takeoff')):
        '''
        '''
        for toff in toffs:
            self.create_phase(slice(toff.slice.start, toff.slice.start + 300))


# TODO: Write some unit tests!
class GoAround5MinRating(FlightPhaseNode):
    '''
    For engines, the period of high power operation is normally 5 minutes from
    the start of takeoff. Also applies in the case of a go-around.
    '''

    def derive(self, gas=S('Go Around And Climbout'), tdwn=S('Touchdown')):
        '''
        We check that the computed phase cannot extend beyond the last
        touchdown, which may arise if a go-around was detected on the final
        approach.
        '''
        for ga in gas:
            startpoint = ga.slice.start
            endpoint = ga.slice.start + 300
            if tdwn:
                endpoint = min(endpoint, tdwn[-1].index)
            if startpoint < endpoint:
                self.create_phase(slice(startpoint, endpoint))


################################################################################


class TaxiIn(FlightPhaseNode):
    """
    This takes the period from start of data to start of takeoff as the taxi
    out, and the end of the landing to the end of the data as taxi in. Could
    be improved to include engines running condition at a later date.
    """
    def derive(self, gnds=S('Mobile'), lands=S('Landing')):
        land = lands.get_last()
        if not land:
            return
        for gnd in gnds:
            if slices_overlap(gnd.slice, land.slice):
                taxi_start = land.slice.stop
                taxi_stop = gnd.slice.stop
                self.create_phase(slice(taxi_start, taxi_stop),
                                  name="Taxi In")


class TaxiOut(FlightPhaseNode):
    """
    This takes the period from start of data to start of takeoff as the taxi
    out, and the end of the landing to the end of the data as taxi in.
    """
    def derive(self, gnds=S('Mobile'), toffs=S('Takeoff')):
        if toffs:
            toff = toffs[0]
            for gnd in gnds:
                if slices_overlap(gnd.slice, toff.slice):
                    taxi_start = gnd.slice.start + 1
                    taxi_stop = toff.slice.start - 1
                    self.create_phase(slice(taxi_start, taxi_stop),
                                      name="Taxi Out")


class Taxiing(FlightPhaseNode):
    def derive(self, t_out=S('Taxi Out'), t_in=S('Taxi In')):
        taxi_slices = slices_or(t_out.get_slices(), t_in.get_slices())
        if taxi_slices:
            self.create_phases(taxi_slices)


class TurningInAir(FlightPhaseNode):
    """
    Rate of Turn is greater than +/- RATE_OF_TURN_FOR_FLIGHT_PHASES (%.2f) in the air
    """ % RATE_OF_TURN_FOR_FLIGHT_PHASES
    def derive(self, rate_of_turn=P('Rate Of Turn'), airborne=S('Airborne')):
        turning = np.ma.masked_inside(repair_mask(rate_of_turn.array),
                                      -RATE_OF_TURN_FOR_FLIGHT_PHASES,
                                      RATE_OF_TURN_FOR_FLIGHT_PHASES)
        turn_slices = np.ma.clump_unmasked(turning)
        for turn_slice in turn_slices:
            if any([is_slice_within_slice(turn_slice, air.slice)
                    for air in airborne]):
                # If the slice is within any airborne section.
                self.create_phase(turn_slice, name="Turning In Air")


class TurningOnGround(FlightPhaseNode):
    """ 
    Turning on ground is computed during the two taxi phases. This\
    avoids\ high speed turnoffs where the aircraft may be travelling at high\
    speed\ at, typically, 30deg from the runway centreline. The landing\
    phase\ turnoff angle is nominally 45 deg, so avoiding this period.
    
    Rate of Turn is greater than +/- RATE_OF_TURN_FOR_TAXI_TURNS (%.2f) on the ground
    """ % RATE_OF_TURN_FOR_TAXI_TURNS
    def derive(self, rate_of_turn=P('Rate Of Turn'), taxi=S('Taxiing')):
        turning = np.ma.masked_inside(repair_mask(rate_of_turn.array),
                                      -RATE_OF_TURN_FOR_TAXI_TURNS,
                                      RATE_OF_TURN_FOR_TAXI_TURNS)
        turn_slices = np.ma.clump_unmasked(turning)
        for turn_slice in turn_slices:
            if any([is_slice_within_slice(turn_slice, txi.slice)
                    for txi in taxi]):
                self.create_phase(turn_slice, name="Turning On Ground")


# NOTE: Python class name restriction: '2DegPitchTo35Ft' not permitted.
class TwoDegPitchTo35Ft(FlightPhaseNode):
    '''
    '''

    name = '2 Deg Pitch To 35 Ft'

    def derive(self, takeoff_rolls=S('Takeoff Roll'), takeoffs=S('Takeoff')):
        for takeoff in takeoffs:
            for takeoff_roll in takeoff_rolls:
                if not is_slice_within_slice(takeoff_roll.slice, takeoff.slice):
                    continue

                if takeoff.slice.stop - takeoff_roll.slice.stop > 1:
                    self.create_section(slice(takeoff_roll.slice.stop, takeoff.slice.stop),
                                    begin=takeoff_roll.stop_edge,
                                    end=takeoff.stop_edge)
                else:
                    self.warning('%s not created as slice less than 1 sample' % self.name)
