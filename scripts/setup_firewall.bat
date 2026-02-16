@echo off
REM Run this as Administrator to configure Windows Firewall

echo Configuring Windows Firewall for Flask API...

REM Add inbound rule for Flask on port 5000
netsh advfirewall firewall add rule name="Flask API Port 5000" dir=in action=allow protocol=TCP localport=5000

REM Verify the rule was added
netsh advfirewall firewall show rule name="Flask API Port 5000"

echo.
echo Firewall configured successfully!
echo Flask API is now accessible on local network at http://YOUR_IP:5000
echo.
echo To find your IP address, run: ipconfig
pause
