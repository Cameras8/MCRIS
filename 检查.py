import psutil

def check_port_usage(port):
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            pid = conn.pid
            if pid:
                process = psutil.Process(pid)
                print(f"Port {port} is used by PID {pid}, process name: {process.name()}")

check_port_usage(1234)
