from cereal import car
from common.params import Params
from openpilot.selfdrive.car import button_pressed
from openpilot.selfdrive.car.chrysler.values import HYBRID_CARS

ButtonType = car.CarState.ButtonEvent.Type

class LongCarController:
  def __init__(self, CP, params, packer):
    self.CP = CP
    self.params = params
    self.packer = packer
    self.last_das_3_counter = -1
    self.last_das_5_counter = -1
    self.hybrid = CP.carFingerprint in HYBRID_CARS

    self.settingsParams = Params()

  def button_control(self, CC, CS):
    if not CS.longControl:
      return False

    if CC.cruiseControl.cancel or button_pressed(CS.out, ButtonType.cancel) or CS.out.brakePressed:
      CS.longEnabled = False
    elif button_pressed(CS.out, ButtonType.accelCruise) or \
        button_pressed(CS.out, ButtonType.decelCruise) or \
        button_pressed(CS.out, ButtonType.resumeCruise):
      CS.longEnabled = True

    accDiff = None
    if button_pressed(CS.out, ButtonType.followInc, False):
      if CC.jvePilotState.carControl.accEco < 2:
        accDiff = 1
    elif button_pressed(CS.out, ButtonType.followDec, False):
      if CC.jvePilotState.carControl.accEco > 0:
        accDiff = -1
    if accDiff is not None:
      newEco = CC.jvePilotState.carControl.accEco + accDiff
      self.settingsParams.put_nonblocking("jvePilot.carState.accEco", str(newEco))

    return True

  def acc(self, longitudinalPlan, frame, CC, CS, can_sends):
    return None
