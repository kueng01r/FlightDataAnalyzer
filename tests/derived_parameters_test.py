try:
    import unittest2 as unittest  # py2.6
except ImportError:
    import unittest
import numpy as np
import mock

import utilities.masked_array_testutils as ma_test
from utilities.struct import Struct
from settings import GRAVITY
from analysis.node import Attribute, A, KPV, KTI, Parameter, P, Section, S
from analysis.flight_phase import Fast, InGroundEffect

from analysis.derived_parameters import (AccelerationVertical,
                                         AirspeedForPhases,
                                         AltitudeAALForPhases,
                                         AltitudeForPhases,
                                         AltitudeRadio,
                                         AltitudeRadioForPhases,
                                         AltitudeTail,
                                         ClimbForPhases,
                                         HeadContinuous,
                                         Pitch,
                                         RateOfClimb,
                                         RateOfClimbForPhases,
                                         RateOfTurn)



class TestAccelerationVertical(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Normal', 'Acceleration Lateral', 
                    'Acceleration Longitudinal', 'Pitch', 'Roll')]
        opts = AccelerationVertical.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_acceleration_vertical_level_on_gound(self):
        # Invoke the class object
        acc_vert = AccelerationVertical(frequency=8)
                        
        acc_vert.derive(
            acc_norm=Parameter('Acceleration Normal',np.ma.ones(8),8),
            acc_lat=Parameter('Acceleration Lateral',np.ma.zeros(4),4),
            acc_long=Parameter('Acceleration Longitudinal',np.ma.zeros(4),4),
            pitch=Parameter('Pitch',np.ma.zeros(2),2),
            roll=Parameter('Roll',np.ma.zeros(2),2))
        
        ma_test.assert_masked_array_approx_equal(acc_vert.array, np.ma.array([1]*8))
        
    def test_acceleration_vertical_pitch_up(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.derive(
            P('Acceleration Normal',np.ma.ones(8)*0.8660254,8),
            P('Acceleration Lateral',np.ma.zeros(4),4),
            P('Acceleration Longitudinal',np.ma.ones(4)*0.5,4),
            P('Pitch',np.ma.ones(2)*30.0,2),
            P('Roll',np.ma.zeros(2),2))

        ma_test.assert_masked_array_approx_equal(acc_vert.array, np.ma.array([1]*8))

    def test_acceleration_vertical_roll_right(self):
        acc_vert = AccelerationVertical(frequency=8)

        acc_vert.derive(
            P('Acceleration Normal',np.ma.ones(8)*0.7071068,8),
            P('Acceleration Lateral',np.ma.ones(4)*(-0.7071068),4),
            P('Acceleration Longitudinal',np.ma.zeros(4),4),
            P('Pitch',np.ma.zeros(2),2),
            P('Roll',np.ma.ones(2)*45,2))

        ma_test.assert_masked_array_approx_equal(acc_vert.array, np.ma.array([1]*8))


class TestAirspeedForPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Airspeed',)]
        opts = AirspeedForPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_airspeed_for_phases_basic(self):
        fast_and_slow = np.ma.array([40,200,190,180,170])
        speed = AirspeedForPhases()
        speed.derive(Parameter('Airspeed', fast_and_slow))
        expected = np.ma.array([40,195,195,185,175])
        ma_test.assert_masked_array_approx_equal(speed.array, expected)

   



class TestAltitudeAALForPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD','Fast')]
        opts = AltitudeAALForPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_altitude_AAL_for_phases_basic(self):
        slow_and_fast_data = np.ma.array(range(60,120,10)+range(120,50,-10))
        up_and_down_data = slow_and_fast_data * 10
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', slow_and_fast_data))
        alt_4_ph = AltitudeAALForPhases()
        alt_4_ph.derive(Parameter('Altitude STD', up_and_down_data), phase_fast)
        expected = np.ma.array([0, 0, 0, 100, 200, 300, 
                                500, 400, 300, 200, 100, 0, 0])
        ma_test.assert_masked_array_approx_equal(alt_4_ph.array, expected)

    def test_altitude_AAL_for_phases_masked_at_lift(self):
        slow_and_fast_data = np.ma.array(range(60,120,10)+range(120,50,-10))
        up_and_down_data = slow_and_fast_data * 10
        up_and_down_data[1:4] = np.ma.masked
        phase_fast = Fast()
        phase_fast.derive(Parameter('Airspeed', slow_and_fast_data))
        alt_4_ph = AltitudeAALForPhases()
        alt_4_ph.derive(Parameter('Altitude STD', up_and_down_data), phase_fast)
        expected = np.ma.array([0, 0, 0, 100, 200, 300, 
                                500, 400, 300, 200, 100, 0, 0])
        ma_test.assert_masked_array_approx_equal(alt_4_ph.array, expected)

        AltitudeRadioForPhases

class TestAltitudeRadio(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio Sensor', 'Pitch',
                     'Main Gear To Altitude Radio')]
        opts = AltitudeRadio.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_altitude_radio(self):
        alt_rad = AltitudeRadio()
        alt_rad.derive(
            Parameter('Altitude Radio Sensor', np.ma.ones(10)*10, 1,0.0),
            Parameter('Pitch', (np.ma.array(range(10))-2)*5, 1,0.0),
            Attribute('Main Gear To Altitude Radio', 10.0)
        )
        result = alt_rad.array
        answer = np.ma.array(data=[11.7364817767,
                                   10.8715574275,
                                   10.0,
                                   9.12844257252,
                                   8.26351822333,
                                   7.41180954897,
                                   6.57979856674,
                                   5.77381738259,
                                   5.0,
                                   4.26423563649],
                             dtype=np.float, mask=False)
        np.testing.assert_array_almost_equal(alt_rad.array, answer)

class TestAltitudeForPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD',)]
        opts = AltitudeForPhases.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_altitude_for_phases_repair(self):
        alt_4_ph = AltitudeForPhases()
        raw_data = np.ma.array([0,1,2])
        raw_data[1] = np.ma.masked
        alt_4_ph.derive(Parameter('Altitude STD', raw_data, 1,0.0))
        expected = np.ma.array([0,0,0],mask=False)
        np.testing.assert_array_equal(alt_4_ph.array, expected)
        
    def test_altitude_for_phases_hysteresis(self):
        alt_4_ph = AltitudeForPhases()
        testwave = np.sin(np.arange(0,6,0.1))*200
        alt_4_ph.derive(Parameter('Altitude STD', np.ma.array(testwave), 1,0.0))

        answer = np.ma.array(data = [0.,0.,0.,0.,0.,0.,12.92849468,28.84353745,
                                     43.47121818,56.66538193,68.29419696,
                                     78.24147201,86.40781719,92.71163708,
                                     97.089946,99.49899732,99.91472061,
                                     99.91472061,99.91472061,99.91472061,
                                     99.91472061,99.91472061,99.91472061,
                                     99.91472061,99.91472061,99.91472061,
                                     99.91472061,99.91472061,99.91472061,
                                     99.91472061,99.91472061,99.91472061,
                                     88.32517131,68.45086117,48.89177959,
                                     29.84335446,11.49591134,-5.96722818,
                                     -22.37157819,-37.55323184,-51.36049906,
                                     -63.65542221,-74.31515448,-83.23318735,
                                     -90.32041478,-95.50602353,-98.73820073,
                                     -99.98465151,-99.98465151,-99.98465151,
                                     -99.98465151,-99.98465151,-99.98465151,
                                     -99.98465151,-99.98465151,-99.98465151,
                                     -99.98465151,-99.98465151,-99.98465151,
                                     -99.98465151],mask = False)
        np.testing.assert_array_almost_equal(alt_4_ph.array, answer)


class TestAltitudeRadioForPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio',)]
        opts = AltitudeRadioForPhases.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_altitude_for_radio_phases_repair(self):
        alt_4_ph = AltitudeRadioForPhases()
        raw_data = np.ma.array([0,1,2])
        raw_data[1] = np.ma.masked
        alt_4_ph.derive(Parameter('Altitude Radio', raw_data, 1,0.0))
        expected = np.ma.array([0,0,0],mask=False)
        np.testing.assert_array_equal(alt_4_ph.array, expected)


class TestAltitudeTail(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude Radio', 'Pitch','Dist Gear To Tail')]
        opts = AltitudeTail.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_altitude_tail(self):
        talt = AltitudeTail()
        talt.derive(Parameter('Altitude Radio', np.ma.ones(10)*10, 1,0.0),
                    Parameter('Pitch', np.ma.array(range(10))*2, 1,0.0),
                    Attribute('Dist Gear To Tail', 35.0)
                    )
        result = talt.array
        # At 35ft and 18deg nose up, the tail just scrapes the runway with 10ft
        # clearance at the mainwheels...
        answer = np.ma.array(data=[10.0,
                                   8.77851761541,
                                   7.55852341896,
                                   6.34150378563,
                                   5.1289414664,
                                   3.92231378166,
                                   2.72309082138,
                                   1.53273365401,
                                   0.352692546405,
                                   -0.815594803123],
                             dtype=np.float, mask=False)
        np.testing.assert_array_almost_equal(result.data, answer.data)


class TestClimbForPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD','Fast')]
        opts = ClimbForPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_climb_for_phases_basic(self):
        up_and_down_data = np.ma.array([0,2,5,3,2,5,6,8])
        phase_fast = Fast()
        phase_fast.derive(P('Airspeed', np.ma.array([100]*8)))
        climb = ClimbForPhases()
        climb.derive(Parameter('Altitude STD', up_and_down_data), phase_fast)
        expected = np.ma.array([0,2,5,0,0,3,4,6])
        ma_test.assert_masked_array_approx_equal(climb.array, expected)

   
'''
class TestFlightPhaseRateOfClimb(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD',)]
        opts = FlightPhaseRateOfClimb.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_flight_phase_rate_of_climb(self):
        params = {'Altitude STD':Parameter('', np.ma.array(range(10))+100)}
        roc = FlightPhaseRateOfClimb()
        roc.derive(P('Altitude STD', np.ma.array(range(10))+100))
        answer = np.ma.array(data=[1]*10, dtype=np.float,
                             mask=False)
        ma_test.assert_masked_array_approx_equal(roc.array, answer)

    def test_flight_phase_rate_of_climb_check_hysteresis(self):
        return NotImplemented
'''

        
class TestHeadContinuous(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Heading Magnetic',)]
        opts = HeadContinuous.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_heading_continuous(self):
        f = HeadContinuous()
        f.derive(P('Heading Magnetic',np.ma.remainder(
            np.ma.array(range(10))+355,360.0)))
        
        answer = np.ma.array(data=[355.0, 356.0, 357.0, 358.0, 359.0, 360.0, 
                                   361.0, 362.0, 363.0, 364.0],
                             dtype=np.float, mask=False)

        #ma_test.assert_masked_array_approx_equal(res, answer)
        np.testing.assert_array_equal(f.array.data, answer.data)
        
        
class TestPitch(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Pitch (1)', 'Pitch (2)')]
        opts = Pitch.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_pitch_combination(self):
        pch = Pitch()
        pch.derive(P('Pitch (1)', np.ma.array(range(5)), 1,0.1),
                   P('Pitch (2)', np.ma.array(range(5))+10, 1,0.6)
                  )
        answer = np.ma.array(data=[0,10,1,11,2,12,3,13,4,14],
                             dtype=np.float, mask=False)
        np.testing.assert_array_equal(pch.array, answer.data)

    def test_pitch_reverse_combination(self):
        pch = Pitch()
        pch.derive(P('Pitch (1)', np.ma.array(range(5))+1, 1,0.75),
                   P('Pitch (2)', np.ma.array(range(5))+10, 1,0.25)
                  )
        answer = np.ma.array(data=[10,1,11,2,12,3,13,4,14,5],
                             dtype=np.float, mask=False)
        np.testing.assert_array_equal(pch.array, answer.data)

    def test_pitch_error_different_rates(self):
        pch = Pitch()
        self.assertRaises(ValueError, pch.derive,
                          P('Pitch (1)', np.ma.array(range(5)), 2,0.1),
                          P('Pitch (2)', np.ma.array(range(10))+10, 4,0.6))
        
    def test_pitch_error_different_offsets(self):
        pch = Pitch()
        self.assertRaises(ValueError, pch.derive,
                          P('Pitch (1)', np.ma.array(range(5)), 1,0.11),
                          P('Pitch (2)', np.ma.array(range(5)), 1,0.6))
        
class TestRateOfClimb(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Acceleration Vertical',
                     'Altitude STD',
                     'Altitude Radio',
                     'In Ground Effect')]
        opts = RateOfClimb.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_rate_of_climb_basic(self):
        az = P('Acceleration Vertical', np.ma.array([1]*10))
        alt_std = P('Altitude STD', np.ma.array([100]*10))
        alt_rad = P('Altitude Radio', np.ma.array([0]*10))
        ige = InGroundEffect()
        ige.derive(alt_rad)
        
        roc = RateOfClimb()
        roc.derive(az, alt_std, alt_rad, ige)

        expected = np.ma.array(data=[0]*10, dtype=np.float,
                             mask=False)
        ma_test.assert_masked_array_approx_equal(roc.array, expected)

    def test_rate_of_climb_bump(self):
        az = P('Acceleration Vertical', np.ma.array([1]*10,dtype=float))
        az.array[2:4] = 1.1
        # (Low acceleration for this test as the sample rate is only 1Hz).
        alt_std = P('Altitude STD', np.ma.array([100]*10))
        alt_rad = P('Altitude Radio', np.ma.array([0]*10))
        ige = InGroundEffect()
        ige.derive(alt_rad)
        
        roc = RateOfClimb()
        roc.derive(az, alt_std, alt_rad, ige)
        expected = np.ma.array(data=[0, 0, 90.491803, 259.890198, 316.826998,
                                     275.171138, 237.858958, 204.464431,
                                     174.602508, 147.925205], mask=False)
        ma_test.assert_masked_array_approx_equal(roc.array, expected)


    def test_rate_of_climb_step_oge(self):
        az = P('Acceleration Vertical', np.ma.array([1]*20,dtype=float))
        alt_std = P('Altitude STD', np.ma.array([100]*20))
        alt_std.array[5:] = 120
        alt_rad = P('Altitude Radio', np.ma.array([100]*20))
        ige = InGroundEffect()
        ige.derive(alt_rad)
        roc = RateOfClimb()
        roc.derive(az, alt_std, alt_rad, ige)
        expected = np.ma.array(data=[0, 0, 0, 0, 0, 1.14285714e+02, 1.03401361e+02,
                                     9.35536119e+01, 8.46437441e+01, 7.65824352e+01,
                                     6.92888699e+01, 6.26899299e+01, 5.67194604e+01,
                                     5.13176070e+01, 4.64302159e+01, 4.20082906e+01,
                                     3.80075010e+01, 3.43877390e+01, 3.11127162e+01,
                                     2.81496004e+01], mask=False)
        ma_test.assert_masked_array_approx_equal(roc.array, expected)


    def test_rate_of_climb_step_ige(self):
        az = P('Acceleration Vertical', np.ma.array([1]*10,dtype=float))
        alt_std = P('Altitude STD', np.ma.array([100]*10))
        alt_rad = P('Altitude Radio', np.ma.array([0]*10))
        alt_rad.array[3:] = 10
        # With an altitude of 0-20ft the aircraft is always in ground effect.        
        ige = InGroundEffect()
        ige.derive(alt_rad)
        
        roc = RateOfClimb()
        roc.derive(az, alt_std, alt_rad, ige)
        expected = np.ma.array(data=[0., 0., 0., 5.71428571e+01, 5.17006803e+01,
                                     4.67768060e+01, 4.23218721e+01, 3.82912176e+01,
                                     3.46444350e+01, 3.13449650e+01], mask=False)
        ma_test.assert_masked_array_approx_equal(roc.array, expected)


    def test_rate_of_climb_combined_signals(self):
        az = P('Acceleration Vertical', np.ma.array([1]*10,dtype=float))
        az.array[2:] += 1/GRAVITY
        slope = (np.cumsum(np.arange(0.0,8.0,1)))
        alt_std = P('Altitude STD', np.ma.array([100]*10,dtype=float))
        alt_std.array[2:] += slope 
        alt_rad = P('Altitude Radio', np.ma.array([0]*10,dtype=float))
        slope *= 1.01 # Just enough to make the values different.
        alt_rad.array[2:] += slope 
        # With an altitude of 0-20ft the aircraft is always in ground effect.        
        ige = InGroundEffect()
        ige.derive(alt_rad)
        
        roc = RateOfClimb()
        roc.derive(az, alt_std, alt_rad, ige)
        expected = np.ma.array(data=[0, 0, 2.81467506e+01, 8.66081933e+01,
                                     1.43457877e+02, 1.98908824e+02, 2.53151819e+02,
                                     3.05601806e+02, 3.57652112e+02, 4.08922826e+02],
                               mask=False)
        ma_test.assert_masked_array_approx_equal(roc.array, expected)


class TestRateOfClimbForPhases(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude STD',)]
        opts = RateOfClimbForPhases.get_operational_combinations()
        self.assertEqual(opts, expected)
        
    def test_rate_of_climb_for_flight_phases_basic(self):
        alt_std = P('Altitude STD', np.ma.arange(10))
        roc = RateOfClimbForPhases()
        roc.derive(alt_std)
        expected = np.ma.array(data=[60]*10, dtype=np.float, mask=False)
        np.testing.assert_array_equal(roc.array, expected)

    def test_rate_of_climb_for_flight_phases_level_flight(self):
        alt_std = P('Altitude STD', np.ma.array([100]*10))
        roc = RateOfClimbForPhases()
        roc.derive(alt_std)
        expected = np.ma.array(data=[0]*10, dtype=np.float, mask=False)
        np.testing.assert_array_equal(roc.array, expected)

        
class TestRateOfTurn(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Head Continuous',)]
        opts = RateOfTurn.get_operational_combinations()
        self.assertEqual(opts, expected)
       
    def test_rate_of_turn(self):
        rot = RateOfTurn()
        rot.derive(P('Head Continuous', np.ma.array(range(10))))
        answer = np.ma.array(data=[1]*10, dtype=np.float)
        np.testing.assert_array_equal(rot.array, answer) # Tests data only; NOT mask
       
    def test_rate_of_turn_phase_stability(self):
        params = {'Head Continuous':Parameter('', np.ma.array([0,0,0,1,0,0,0], 
                                                               dtype=float))}
        rot = RateOfTurn()
        rot.derive(P('Head Continuous', np.ma.array([0,0,0,1,0,0,0],
                                                          dtype=float)))
        answer = np.ma.array([0,0,0.5,0,-0.5,0,0])
        ma_test.assert_masked_array_approx_equal(rot.array, answer)