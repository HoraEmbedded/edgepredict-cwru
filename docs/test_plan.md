# Test Plan and Results

## Test 1: Class detection across files

Each CWRU file was used as the inference source and the predicted class was verified.

- 97.mat  -> Normal
- 105.mat -> InnerRace
- 118.mat -> Ball
- 130.mat -> OuterRace

Result: all classes correctly detected.

## Test 2: CPU and memory usage

Measured with htop during steady state inference on Raspberry Pi 4.

- CPU usage: around 4 to 9 percent
- Memory usage: around 162 MB for the Python process
- Load average: below 1.0 on a 4-core system
- Uptime stable over 18 minutes of continuous inference

## Test 3: Network reconnection

The dashboard recovered automatically after a 10 second network disconnection.

## Test 4: Full reboot

The systemd service started automatically after reboot.
The dashboard was reachable without manual intervention.
