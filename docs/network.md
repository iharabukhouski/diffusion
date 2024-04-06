# host-to-host bandwidth

you need to make sure that 5201 port is allowed by the firewall

```bash
apt install iperf3
ifconfig # gets you internal IP
iperf3 -i 10 -V -s # server
iperf3 -i 10 -V -c <SERVER_IP> # client
```

# host-to-internet bandwidth
```bash
apt install speedtest-cli
speedtest-cli
```

# scan ports

```bash
apt install nmap
nmap -Pn INSTANCE-IP-ADDRESS # from local machine
```
