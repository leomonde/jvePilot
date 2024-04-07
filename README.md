[![openpilot on the comma 3X](https://github.com/commaai/openpilot/assets/8762862/f09e6d29-db2d-4179-80c2-51e8d92bdb5c)](https://comma.ai/shop/comma-3x)

![](https://i.imgur.com/b0ZyIx5.jpg)
# jvePilot Hybrid OpenPilot/ACC for Chrysler/Jeep 
This fork is only for Chrysler/Jeep vehicles and requires a Comma 3 or later device to run. Comma 2 users need to use [this branch](https://github.com/j-vanetten/openpilot/tree/jvePilot-c2-release)

[![Buy me a beer!](https://github.com/j-vanetten/openpilot/blob/jvePilot-release/.github/ButMeABeer.png?raw=true)](https://www.buymeacoffee.com/jvePilot)

Come join us on [Discord](https://discord.gg/r8yaDBdnwH)! 

# Table of contents
- [**Safety Notes**](#safety-notes)
- [jvePilot](#jvepilot)
  * [What is this Fork?](#what-is-this-fork)
  * [Benefits of jvePilot](#benefits-of-jvepilot)
    + [Longitudinal control](#longitudinal-control)
    + [Auto Resume](#auto-resume)
    + [Auto Follow](#auto-follow)
    + [ACC Eco](#acc-eco)
  * [How to use it](#how-to-use-it)
    + [Where to look when setting ACC speed](#where-to-look-when-setting-acc-speed)
- [Install](#install)
  * [Branches](#branches)
- [Customizing](#customizing)
- [Advanced](#advanced-settings)

# **Safety Notes**
* This is my experimental branch, so I'm not responsible for any damage this may cause.
* jvePilot still does not have direct control of the gas and brakes!
  Changing the ACC speed does not always result in the vehicle braking unless the difference in speed is large enough.
  If the speed difference is small, the vehicle just lets off the gas.
* ACC can't go slower that 20mph
* ACC doesn't do a good job at seeing things that are already stopped

---

# jvePilot 
I have a 2018 Grand Cherokee Trailhawk, so I'm only able to confirm features using this vehicle.
* 2017 Gas Chrysler Pacifica: Confirmed by @debugged-hosting

## What is this Fork?
This is my personal OpenPilot fork that includes features that I feel make it a better driving experience for me and possibly others.

## Benefits of jvePilot
* Smother driving in traffic as jvePilot will do a better job at predicting traffic and adjust ACC speed accordingly
* ACC braking by setting the cruse speed lower that the target to help slow the vehicle sooner
* Slow for cars cutting in before ACC does
* Slow in a turn, so you don't have to change the set speed yourself (Speeds are configurable)
* Always On Lateral Control so steering is active even if ACC isn't
* Auto resume after ACC comes to a stop behind vehicle (Can be disabled)
* Auto follow feature to adjust the follow distance based on speed (Speeds are configurable)
* ACC Eco to limit the throttle when accelerating  
* Syncs jvePilot display speed with the vehicle speedometer 
* Use LKAS button in the dash to toggle Experimental Mode if Experimental Mode is enabled in the settings
* Gas/brake indication using green/red colors on speed indicator

### Longitudinal control
This fork combines the speed control logic of OpenPilot with the vehicles Adaptive Cruse Control (ACC).
It does this by changing the ACC speed to match the value OpenPilot calculates as the desired speed.
This brings some of OpenPilots longitudinal control to these vehicles.
Including things like slowing while cornering and slowing when it detects cut-ins.
It will also smooth the braking of ACC when driving in traffic.

On FCA vehicles, only the steering is controlled by jvePilot and speed is left up to the ACC of the vehicle.
This fork takes control of the ACC speed setting and adjusts the ACC speed to match the speed jvePilot would be targeting if it actually was able to control the gas and brakes.
It does this by simulating ACC+ and ACC- button presses on the steering wheel to change the ACC speed.
It is limited as ACC only goes down to 20 mph, so it doesn't help as low speeds.

### Always On Lateral Control
Always On Lateral Control, or AOLC, is a feature that allows jvePilot to steer even if ACC isn't actively engaged.  
Enabling this feature will allow a driver to take control of gas/brakes while still retaining the automatic steering provided by jvePilot.
Because the low torque applied to the steering, it's very easy to override the steering if needed.

### Auto Resume
ACC will come to a stop behind vehicles, however, if stopped too long, it will either stay stopped until resume is pressed, or simply disengage ACC altogether.  
For the case where ACC simply cancels, the driver has to press and hold the brake to keep the vehicle stopped.
Auto resume makes life easier by resuming ACC when the vehicle in front of you begin to move, or, you let off the brake after coming to a standstill.
While stopped, you can still disengage jvePilot by pressing the Cancel button. 

### Auto Follow
Auto Follow is a way to automate the changing of the stock follow distance setting.
It sets the follow distance to closer at slow speeds and increases it the faster you go.
Setting the follow speed to one/two bars helps with keeping up with cars that take off when stopped or at slow speeds.
The faster you go, the more distance you want, so you can have more confidence in ACC being able to stop in case it needs to.

The current enabled state of Auto Follow is as an icon above the ACC Eco button on the jvePilot display.
Pressing Follow + or - will disable Auto Follow giving you full control to set the follow distance. 
To re-enable Auto Follow, hold either Follow + or - for half a second. 
 
### ACC Eco
When enabled, jvePilot will limit how far ahead the ACC setting is above the current speed.  
This prevents the vehicle from using an aggressive throttle to get up to speed saving on gas/battery.

The ACC Eco button is located in the lower right corner of the display.  
Tapping the button cycles between off, level 1, and level 2 eco settings.
Level 2 provides the slowest acceleration and is selected when both leaves are green.    
Level 1 should provide a balance is selected when only one leaf is green.
If you feel these settings are not right for you or your vehicle, see the [ACC Eco](#acc-eco) setting to adjust them. 
Much like your vehicles eco/sport modes, the current setting is persisted between drives.

## How to use it 
When using this branch, you will be setting the max ACC speed on the jvePilot display instead of the one in the dashboard.
jvePilot will then set the ACC setting in the dashboard to the targeted speed, but never exceeding the max speed set on the jvePilot display.
A quick press of the ACC+ and ACC- buttons will change this speed by 5 mph on the jvePilot display, while a long deliberate press (about a 1/2 second press) changes it by 1 mph.
DO NOT hold the ACC+ or ACC- buttons for longer that a 1 second. Either make quick or long deliberate presses only.

### Where to look when setting ACC speed
Do not look at the dashboard when setting your ACC max speed.
Instead, only look at the one on the jvePilot display.
The reason you need to look at jvePilot is because jvePilot will be changing the one in the dashboard.
It will be adjusting it as needed, never raising it above the one set on the jvePilot display.

**ONLY look at the MAX speed on jvePilot when setting the ACC speed instead of the dashboard!**

---

# Install

The easiest way to install jvePilot is to factory reset your [C3](https://www.youtube.com/watch?v=gNnRmEyVSVQ) and use this Custom Software URL: `https://bit.ly/jvePilot-release`

## Branches
`/jvePilot-release` - The latest release.  Will contain the latest version I feel is ready for daily use. Custom Software URL: `https://bit.ly/jvePilot-release`

`/jvePilot-beta` - Sometimes I have people wanting to beta test jvePilot's new features.  Mostly stable, but still can be buggy. Custom Software URL: `https://bit.ly/jvePilot-beta` 

`/feature/*` - These branches are where I'm working on new features.  These are never safe to run as they change all the time.

---
# Customizing
Customizing features and parameters can be done on the UI display.  
Click the gear icon and then select jvePilot from the sidebar.  
Clicking on the text of feature of will show more information about it and allow customization of the feature.
Note that some settings do require a vehicle restart, and some don't 
The settings that don't require a restart may take upward of 5 seconds to take effect after changing them. 


## Slow in Curves
jvePilot will slow in curves so that you don't have to.
* Default: On
* Vehicle Restart Required: No
### Speed Ratio 
Use this to tune the speed in curves to you liking.
Setting this to 1.2 will cause jvePilot to drive 20% faster in turns than if it was set to the default of 1.0
* Default: 1.0
* Units: Ratio
* Vehicle Restart Required: No
* Min/Max values (0.1, 2)
### Drop off
Adjusts how much the speed drops as the curve increases.
Decrease this value to lessen the amount of drop off as the curve increases.
Changing this value will likely require adjusting the Speed Ratio to compensate.
* Default: 2.0
* Vehicle Restart Required: No
* Min/Max values (1.0, 3.0)

## Reverse ACC +/- Speeds
Reverse the stock ACC +/- button's 1mph on short press and 5mph on long press.  Turn off to return to stock style.
* Default: On
* Vehicle Restart Required: Yes

## Auto Resume
This feature allows jvePilot to auto resume from an ACC stop.
* Default: On
* Vehicle Restart Required: Yes

## Auto Follow
If you don't want auto follow enabled on every start, turn this off.
### 1-2 Bar Change Over
When your speed (in MPH) is below this setting, Auto Follow will set the follow setting to one bar.  
When you reach this speed (in MPH), Auto Follow will set the follow setting to two bars.
* Default: 15
* Units: MPH
* Vehicle Restart Required: No
* Min/Max values (0, 300)
### 2-3 Bar Change Over
When your speed (in MPH) is below this setting, Auto Follow will set the follow setting to two bars.
When you reach this speed (in MPH), Auto Follow will set the follow setting to three bars.
* Default: 30
* Units: MPH
* Vehicle Restart Required: No
* Min/Max values (0, 300)
### 3-4 Bar Change Over
When your speed (in MPH) is below this setting, Auto Follow will set the follow setting to three bars.
When you reach this speed (in MPH), Auto Follow will set the follow setting to four bars.
* Default: 65
* Units: MPH
* Vehicle Restart Required: No
* Min/Max values (0, 300)

## ACC Eco
ACC Eco limits acceleration by keep the ACC cruise speed closer to your current speed.
These setting are how far ahead, in MPH, of your current speed ACC will be set.  
The higher the number, the more aggressive ACC will be when accelerating.
### Keep ahead at ACC Eco level 1
Use this setting to adjust ACC Eco level 1 (one green leaf) for a balance of speed and eco-ness  
* Default: 7
* Units: MPH
* Vehicle Restart Required: No
* Min/Max values 1, 100
### Keep ahead at ACC Eco level 2
Use this setting to adjust ACC Eco level 2 (two green leaves) for maximum eco-ness
* Default: 5
* Units: MPH
* Vehicle Restart Required: No
* Min/Max values 1, 100

## No steer alert 
When this is enabled, you will hear a chime when your vehicle drops to a certain speed and can no longer be steered.
* Default: On
* Vehicle Restart Required: No

## jvePilot Control Settings
### Device Offset
Compensate for mounting your device off center in the windshield.
If you mounted your device off center, use this setting to compensate.
Use positive values if your device is to the left of center and negative if it's to the right.
* Default: 0
* Units: Meters
* Vehicle Restart Required: No
* Min/Max values -1, 1

# Always On Lateral Control
When enabled, jvePilot will continue to steer without having to engage ACC.  
You will still need to enable ACC using the ACC button on the steering wheel. 
* Default: False 
* Vehicle Restart Required: No

# Advanced settings
These settings are for advanced users doing advanced things. 

### Minimum Steer Check  
When disabled, jvePilot will no longer put a minimum on steer speed.
Requires a mod like a [hardware interceptor](https://github.com/xps-genesis/panda/tree/xps_wp_chrysler_basic).
* Default: True 
* Vehicle Restart Required: Yes

### Vision Only
When enabled, the model will no longer use any radar signals and rely on vision only.
Enable this setting if you are seeing the lead car yellow triangle acting erratically.  
* Default: False 
* Vehicle Restart Required: Yes

---


What is openpilot?
------

[openpilot](http://github.com/commaai/openpilot) is an open source driver assistance system. Currently, openpilot performs the functions of Adaptive Cruise Control (ACC), Automated Lane Centering (ALC), Forward Collision Warning (FCW), and Lane Departure Warning (LDW) for a growing variety of [supported car makes, models, and model years](docs/CARS.md). In addition, while openpilot is engaged, a camera-based Driver Monitoring (DM) feature alerts distracted and asleep drivers. See more about [the vehicle integration](docs/INTEGRATION.md) and [limitations](docs/LIMITATIONS.md).

<table>
  <tr>
    <td><a href="https://youtu.be/NmBfgOanCyk" title="Video By Greer Viau"><img src="https://github.com/commaai/openpilot/assets/8762862/2f7112ae-f748-4f39-b617-fabd689c3772"></a></td>
    <td><a href="https://youtu.be/VHKyqZ7t8Gw" title="Video By Logan LeGrand"><img src="https://github.com/commaai/openpilot/assets/8762862/92351544-2833-40d7-9e0b-7ef7ae37ec4c"></a></td>
    <td><a href="https://youtu.be/SUIZYzxtMQs" title="A drive to Taco Bell"><img src="https://github.com/commaai/openpilot/assets/8762862/05ceefc5-2628-439c-a9b2-89ce77dc6f63"></a></td>
  </tr>
</table>

To start using openpilot in a car
------

To use openpilot in a car, you need four things:
1. **Supported Device:** a comma 3/3X, available at [comma.ai/shop](https://comma.ai/shop/comma-3x).
2. **Software:** The setup procedure for the comma 3/3X allows users to enter a URL for custom software. Use the URL `openpilot.comma.ai` to install the release version.
3. **Supported Car:** Ensure that you have one of [the 250+ supported cars](docs/CARS.md).
4. **Car Harness:** You will also need a [car harness](https://comma.ai/shop/car-harness) to connect your comma 3/3X to your car.

We have detailed instructions for [how to install the harness and device in a car](https://comma.ai/setup). Note that it's possible to run openpilot on [other hardware](https://blog.comma.ai/self-driving-car-for-free/), although it's not plug-and-play.

To start developing openpilot
------

openpilot is developed by [comma](https://comma.ai/) and by users like you. We welcome both pull requests and issues on [GitHub](http://github.com/commaai/openpilot).

* Join the [community Discord](https://discord.comma.ai)
* Check out [the contributing docs](docs/CONTRIBUTING.md)
* Check out the [openpilot tools](tools/)
* Read about the [development workflow](docs/WORKFLOW.md)
* Code documentation lives at https://docs.comma.ai
* Information about running openpilot lives on the [community wiki](https://github.com/commaai/openpilot/wiki)

Want to get paid to work on openpilot? [comma is hiring](https://comma.ai/jobs#open-positions) and offers lots of [bounties](docs/BOUNTIES.md) for external contributors.

Safety and Testing
----

* openpilot observes [ISO26262](https://en.wikipedia.org/wiki/ISO_26262) guidelines, see [SAFETY.md](docs/SAFETY.md) for more details.
* openpilot has software-in-the-loop [tests](.github/workflows/selfdrive_tests.yaml) that run on every commit.
* The code enforcing the safety model lives in panda and is written in C, see [code rigor](https://github.com/commaai/panda#code-rigor) for more details.
* panda has software-in-the-loop [safety tests](https://github.com/commaai/panda/tree/master/tests/safety).
* Internally, we have a hardware-in-the-loop Jenkins test suite that builds and unit tests the various processes.
* panda has additional hardware-in-the-loop [tests](https://github.com/commaai/panda/blob/master/Jenkinsfile).
* We run the latest openpilot in a testing closet containing 10 comma devices continuously replaying routes.

User Data and comma Account
------

By default, openpilot uploads the driving data to our servers. You can also access your data through [comma connect](https://connect.comma.ai/). We use your data to train better models and improve openpilot for everyone.

openpilot is open source software: the user is free to disable data collection if they wish to do so.

openpilot logs the road-facing cameras, CAN, GPS, IMU, magnetometer, thermal sensors, crashes, and operating system logs.
The driver-facing camera is only logged if you explicitly opt-in in settings. The microphone is not recorded.

By using openpilot, you agree to [our Privacy Policy](https://comma.ai/privacy). You understand that use of this software or its related services will generate certain types of user data, which may be logged and stored at the sole discretion of comma. By accepting this agreement, you grant an irrevocable, perpetual, worldwide right to comma for the use of this data.

Licensing
------

openpilot is released under the MIT license. Some parts of the software are released under other licenses as specified.

Any user of this software shall indemnify and hold harmless Comma.ai, Inc. and its directors, officers, employees, agents, stockholders, affiliates, subcontractors and customers from and against all allegations, claims, actions, suits, demands, damages, liabilities, obligations, losses, settlements, judgments, costs and expenses (including without limitation attorneys’ fees and costs) which arise out of, relate to or result from any use of this software by user.

**THIS IS ALPHA QUALITY SOFTWARE FOR RESEARCH PURPOSES ONLY. THIS IS NOT A PRODUCT.
YOU ARE RESPONSIBLE FOR COMPLYING WITH LOCAL LAWS AND REGULATIONS.
NO WARRANTY EXPRESSED OR IMPLIED.**

---

<img src="https://d1qb2nb5cznatu.cloudfront.net/startups/i/1061157-bc7e9bf3b246ece7322e6ffe653f6af8-medium_jpg.jpg?buster=1458363130" width="75"></img> <img src="https://cdn-images-1.medium.com/max/1600/1*C87EjxGeMPrkTuVRVWVg4w.png" width="225"></img>

![openpilot tests](https://github.com/commaai/openpilot/actions/workflows/selfdrive_tests.yaml/badge.svg)
[![codecov](https://codecov.io/gh/commaai/openpilot/branch/master/graph/badge.svg)](https://codecov.io/gh/commaai/openpilot)
