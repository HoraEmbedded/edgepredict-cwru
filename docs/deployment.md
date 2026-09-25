# Deployment on Raspberry Pi 4


## Environment

- Hostname: edgepredictpi
- User: horaembedded
- Project path: /home/horaembedded/edgepredict-cwru
- Python virtual environment: /home/horaembedded/edgepredict-cwru/venv

## Systemd service

The application runs as a systemd service named edgepredictpi.

Service file location on the Pi:

/etc/systemd/system/edgepredictpi.service

A copy is stored in the repository at systemd/edgepredictpi.service.

## Commands

Enable at boot:

    sudo systemctl enable edgepredictpi.service

Start now:

    sudo systemctl start edgepredictpi.service

Check status:

    sudo systemctl status edgepredictpi.service

Stop:

    sudo systemctl stop edgepredictpi.service

View logs:

    journalctl -u edgepredictpi.service -f

## Access

The dashboard is reachable from any device on the same network:

    http://<raspberry-pi-ip>:5000

Example:

    http://192.168.1.18:5000

Note: hostname.local (mDNS) may not resolve on all networks.
Using the IP address directly is more reliable.
