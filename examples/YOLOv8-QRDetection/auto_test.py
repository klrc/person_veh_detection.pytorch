import serial
import multiprocessing
import time
from loguru import logger

READY_FLAG = "/serial_ready_flag"
STOP_FLAG = "/serial_stop_flag"
SILENT_FLAG = "/server_silent_flag"
DEAD_FLAG = "/server_dead_flag"

CTRL_C = chr(0x03)


def listen_serial(port, baudrate, input: multiprocessing.Queue, output: multiprocessing.Queue):
    # 尝试连接串口
    logger.debug(f"Connecting to serial port: {port}")
    conn = serial.Serial(port, baudrate, timeout=0.1)
    if not conn.is_open:
        logger.error(f"Unable to open serial port: {port}")
        return None
    logger.success(f"Connected to serial port: {port}")
    output.put(READY_FLAG)

    silent = False
    while True:
        # 读取数据（串口->output)
        try:
            response = conn.readline().decode("utf-8")  # 读取串口数据并解码为字符串
            if response.strip() != "":
                if not silent:
                    print(response, end="")  # 打印串口数据
                output.put(response.strip())  # 将串口数据放入输出队列
        except UnicodeDecodeError:  # 捕获Unicode解码错误
            pass
        except serial.SerialException:  # 捕获串口异常 (XC01特有问题)
            pass

        # 写入数据 (input->串口)
        if input.empty():
            continue
        line = input.get()
        if line == SILENT_FLAG:
            silent = True
            logger.debug("Switch to silent mode")
            continue
        if line == STOP_FLAG:
            break
        try:
            conn.write(line + b"\n")  # 发送数据到串口
            logger.debug(f"Write line: {line}")
        except EOFError:  # 捕获文件结束错误
            pass

    # 关闭串口
    logger.debug("Stop flag received")
    conn.close()
    logger.success(f"Serial port {port} is closed.")  # 打印串口关闭信息
    output.put(DEAD_FLAG)
    logger.debug("Send DEAD_FLAG")


class MySerial:
    def __init__(self, port="/dev/ttyUSB2", baudrate=115200) -> None:
        self.port = port
        self.baudrate = baudrate
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.process = None
        self.is_dead = False
        self.verbose = True

    def __enter__(self):
        self.process = multiprocessing.Process(target=listen_serial, args=(self.port, self.baudrate, self.input_queue, self.output_queue))  # 创建监听串口的进程
        self.process.daemon = True  # 设置进程为守护进程
        self.process.start()  # 启动进程
        ret = self.wait_pattern(READY_FLAG, timeout=1)  # 等待正确连接
        if not ret:
            raise Exception(f"Unable to connect serial {self.port}@{self.baudrate}")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        logger.debug(f"Exiting serial port: {self.port}")
        self.send_line(CTRL_C)
        self.send_flag(STOP_FLAG)
        logger.debug("Waiting for process to join...")
        self.process.join()  # 等待串口服务关闭

    def send_flag(self, flag):
        self.input_queue.put(flag)

    # 写入一行数据
    def send_line(self, line=""):
        self.input_queue.put(line.encode("utf-8"))

    # 等待指定时间
    def wait(self, timeout=100):
        logger.debug(f"Waiting for {timeout} seconds...")  # 打印等待信息
        time.sleep(timeout)  # 休眠指定时间

    # 检查消息队列中是否包含指定模式的消息
    def get_line(self, timeout=-1, check_interval=0.5):
        while True:
            if self.is_dead:
                break
            elif self.output_queue.empty():  # 如果消息队列为空
                if timeout > 0:
                    time.sleep(check_interval)  # 休眠指定时间
                    timeout = max(timeout - check_interval, 0)
                elif timeout == -1:
                    time.sleep(check_interval)  # 休眠指定时间
                else:
                    return None
            else:
                message = self.output_queue.get()  # 获取消息队列中的消息
                if message == DEAD_FLAG:
                    self.is_dead = True
                    logger.debug(f"DEAD_FLAG received, set self.is_dead={self.is_dead}")
                else:
                    return message

    # 获取数据以分析
    def collect(self, stop_pattern, timeout=-1, check_interval=0.5, strict=True):
        ret = []
        if self.verbose:
            logger.debug(f"Collect till stop pattern: {stop_pattern}")
        while True:
            if self.is_dead:
                break
            elif self.output_queue.empty():  # 如果消息队列为空
                if timeout > 0:
                    time.sleep(check_interval)  # 休眠指定时间
                    timeout = max(timeout - check_interval, 0)
                elif timeout == -1:
                    time.sleep(check_interval)  # 休眠指定时间
                else:
                    if strict:
                        raise AssertionError(f"Pattern {stop_pattern} not found within timeout period.")
                    else:
                        logger.warning(f"Pattern {stop_pattern} not found within timeout period.")  # 打印超时警告信息
                    break
            else:
                message = self.output_queue.get()  # 获取消息队列中的消息
                if message == DEAD_FLAG:
                    self.is_dead = True
                    logger.debug(f"DEAD_FLAG received, set self.is_dead={self.is_dead}")
                else:
                    ret.append(message)
                    if stop_pattern in message:  # 如果指定模式在消息中
                        if self.verbose:
                            logger.success(f"Pattern found: {message}")  # 打印找到指定模式的消息信息
                        break
        return ret

    # 检查消息队列中是否包含指定模式的消息
    def wait_pattern(self, pattern, timeout=-1, check_interval=0.5, strict=True):
        message = self.collect(pattern, timeout, check_interval, strict)
        return len(message) > 0 and pattern in message[-1]

    # 清空消息队列
    def clear(self):
        while not self.output_queue.empty():  # 当消息队列不为空时
            self.output_queue.get()  # 取出消息队列中的消息
        logger.success("Message queue cleared.")  # 打印清空消息队列信息

    def silent(self):
        self.send_flag(SILENT_FLAG)
        self.verbose = False


if __name__ == "__main__":
    with MySerial("/dev/ttyUSB0", 115200) as self:
        # self.send_line("reboot")
        # self.wait_pattern("Processing /etc/profile... Done")
        self.send_line(CTRL_C)
        self.send_line(CTRL_C)
        if not self.wait_pattern(" #", timeout=5, strict=False):
            self.wait_pattern("Processing /etc/profile... Done")
        self.clear()

        self.send_line("lsmod")
        if not self.wait_pattern("nnp", timeout=1, strict=False):
            self.send_line("ifconfig eth0 hw ether 02:a2:72:a3:a4:a5")
            self.send_line("ifconfig eth0 192.168.1.100 netmask 255.255.255.0 up")
            self.send_line("mount -n -o sync,noac,nolock 192.168.1.200:/nfs /nfs")
            self.send_line("telnetd")

            self.send_line("cd /nfs/opt/molchip/sdk/XC01_SDK_nightly && sh loadko_xc01_6x8_ddr2.sh")
            self.wait_pattern("current clock rate setting", timeout=100)
        self.clear()

        self.send_line("cd /nfs/projects/ultralytics-yolov8-obb/examples/YOLOv8-QRDetection")
        self.wait_pattern(" #")
        self.clear()

        self.send_line("./tools/benchmark_test -s 640x640 -f nv12 -m yolov8n-obb_nnp310_combine.ty")
        message = self.collect(stop_pattern="<test finished>")
        self.clear()
