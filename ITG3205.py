import serial
import datetime
import math
import numpy as np
from multiprocessing import Process, Queue


class Point:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return 'x:%.2f y:%.2f z:%.2f' % (self.x, self.y, self.z)


(Angle, Acc, Gyro) = (Point,) * 3


class ITG3205:
    IMU_KP = 1.5
    IMU_KI = 0.0005

    def __init__(self, input_q):
        self.input_q = input_q
        self.filter_cnt = 0
        self.FILTER_LENGTH = 20
        self.acc_buff = [Acc] * self.FILTER_LENGTH
        (self.acc_offset_x, self.acc_offset_y, self.acc_offset_z) = (0,) * 3
        (self.gyro_offset_x, self.gyro_offset_y, self.gyro_offset_z) = (0,) * 3

        self.his_q0, self.his_q1, self.his_q2, self.his_q3 = (1.0, 0.0, 0.0, 0.0)
        self.angle = Angle()
        self.v = np.array([1.0, 0.0, 0.0])
        # x、y、z轴的比例误差
        self.ex_int, self.ey_int, self.ez_int = (0.0, 0.0, 0.0)
        self.frame_len = 80
        (self.pre_time, self.delta_time) = (datetime.datetime.now(), None)
        self.filled = False
        self.data_offset(100)
        self.fill_acc_buff()

    def fill_acc_buff(self):
        while not self.filled:
            data_frame = self.read_data()
            self.acc_filter(self.get_acc(data_frame))

    @staticmethod
    def bytes2int(x):
        return int.from_bytes(x, byteorder='big', signed=True)

    @staticmethod
    def get_angle(x, y):
        lx = np.sqrt(x.dot(x))
        ly = np.sqrt(y.dot(y))
        cos_angle = x.dot(y) / (lx * ly)
        res = np.arccos(cos_angle)
        return res * 180 / np.pi

    def get_gyro(self, data):
        # 温度buff
        b2i = ITG3205.bytes2int
        temp = b2i(data[7] + data[6])
        # 三轴角速度buff
        x = b2i(data[1] + data[0]) - self.gyro_offset_x
        y = b2i(data[3] + data[2]) - self.gyro_offset_y
        z = b2i(data[5] + data[4]) - self.gyro_offset_z
        # print(x, y, z)
        return x, y, z, temp

    def angular_velocity(self, data):
        x, y, z, temp = self.get_gyro(data)
        # print('x: %d y: %d z: %d' % (x, y, z))
        # 设置三轴角速度阈值，防止静止状态下漂移
        # threshold = 5
        # if -threshold <= x <= threshold:
        #     x = 0
        # if -threshold <= y <= threshold:
        #     y = 0
        # if -threshold <= z <= threshold:
        #     z = 0
        temp = 35 + (1.0 * temp + 13200) / 280
        x = x / 14.375 * math.pi / 180
        y = y / 14.375 * math.pi / 180
        z = z / 14.375 * math.pi / 180
        return x, y, z, temp

    def get_acc(self, data):
        b2i = ITG3205.bytes2int
        ax = b2i(data[-5] + data[-6]) - self.acc_offset_x
        ay = b2i(data[-3] + data[-4]) - self.acc_offset_y
        az = b2i(data[-1] + data[-2]) - self.acc_offset_z
        return ax, ay, az

    def acc_filter(self, tup):
        ax, ay, az = tup
        # global filter_cnt, filled, acc_buff
        self.acc_buff[self.filter_cnt] = Acc(ax, ay, az)
        self.filter_cnt += 1
        if self.filter_cnt >= self.FILTER_LENGTH:
            self.filter_cnt = 0
            self.filled = True

        length = self.FILTER_LENGTH if self.filled else self.filter_cnt

        tsum = Acc()
        for i in range(length):
            tsum.x += self.acc_buff[i].x
            tsum.y += self.acc_buff[i].y
            tsum.z += self.acc_buff[i].z

        return tsum.x / length, tsum.y / length, tsum.z / length

    def acc(self, data):
        ax, ay, az = self.acc_filter(data)
        # pitch = math.atan2(ay, math.sqrt(ax * ax + az * az)) * 180 / math.pi
        # roll = math.atan2(-ax, az) * 180 / math.pi
        # print(pitch, roll, end='\r')
        ax = 0.039 * ax
        ay = 0.039 * ay
        az = 0.039 * az
        # print("ax: %.2f ay: %.2f az: %.2f" % (ax, ay, az))
        return ax, ay, az

    def imu_update(self, gx, gy, gz, ax, ay, az, output_q):
        half_T = self.delta_time / 2

        if ax * ay * az == 0:
            print('ax * ay * az is 0')
            return

        norm = math.sqrt(ax * ax + ay * ay + az * az)
        ax /= norm
        ay /= norm
        az /= norm

        vx = (self.his_q1 * self.his_q3 - self.his_q0 * self.his_q2) * 2
        vy = (self.his_q0 * self.his_q1 + self.his_q2 * self.his_q3) * 2
        vz = self.his_q0 * self.his_q0 - self.his_q1 * self.his_q1 - self.his_q2 * self.his_q2 + self.his_q3 * self.his_q3

        # 向量外积再相减得到差分就是误差，两个单位向量的差积即为误差向量
        ex = (ay * vz - az * vy)
        ey = (az * vx - ax * vz)
        ez = (ax * vy - ay * vx)

        # 对误差进行PI计算
        self.ex_int = self.ex_int + ex * ITG3205.IMU_KI
        self.ey_int = self.ey_int + ey * ITG3205.IMU_KI
        self.ez_int = self.ez_int + ez * ITG3205.IMU_KI

        gx = gx + ITG3205.IMU_KP * ex + self.ex_int
        gy = gy + ITG3205.IMU_KP * ey + self.ey_int
        gz = gz + ITG3205.IMU_KP * ez + self.ez_int

        # 四元素的微分方程
        q0 = self.his_q0 + (-self.his_q1 * gx - self.his_q2 * gy - self.his_q3 * gz) * half_T
        q1 = self.his_q1 + (self.his_q0 * gx + self.his_q2 * gz - self.his_q3 * gy) * half_T
        q2 = self.his_q2 + (self.his_q0 * gy - self.his_q1 * gz + self.his_q3 * gx) * half_T
        q3 = self.his_q3 + (self.his_q0 * gz + self.his_q1 * gy - self.his_q2 * gx) * half_T

        q0q0 = q0 * q0
        q0q1 = q0 * q1
        q0q2 = q0 * q2
        q0q3 = q0 * q3
        q1q1 = q1 * q1
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q2q2 = q2 * q2
        q2q3 = q2 * q3
        q3q3 = q3 * q3

        # 规范化Pitch、Roll轴四元数
        norm = math.sqrt(q0q0 + q1q1 + q2q2 + q3q3)
        q0 = q0 / norm
        q1 = q1 / norm
        q2 = q2 / norm
        q3 = q3 / norm

        # 放在b系中算角度
        # tv = np.array([2 * (q1q2 + q0q3), 1 - 2 * (q1q1 + q3q3), 2 * (q2q3 - q0q1)])

        # 放在n系中算角度，用于多陀螺仪情况
        # 起始向量(1, 0, 0)
        tv2 = np.array([1 - 2 * (q2q2 + q3q3), 2 * (q1q2 + q0q3), 2 * (q1q3 - q0q2)])
        # 起始向量(0, 0, 1)
        # tv2 = np.array([2 * (q1q3 + q0q2), 2 * (q2q3 - q0q1), 1 - 2 * (q1q1 + q2q2)])
        # 起始向量(0, 1, 0)
        # tv2 = np.array([2 * (q1q2 - q0q3), 1 - 2 * (q1q1 + q3q3), 2 * (q2q3 + q0q1)])

        # h_tv2 = tv2.copy()
        # h_tv2[2] = 0.0
        # v_tv2 = tv2.copy()
        # v_tv2[1] = 0.0

        # 求解欧拉角
        self.angle.x = math.atan2(2 * q2q3 + 2 * q0q1, -2 * q1q1 - 2 * q2q2 + 1)
        self.angle.y = math.asin(-2 * q1q3 + 2 * q0q2)
        self.angle.z = math.atan2(2 * q1q2 + 2 * q0q3, -2 * q2q2 - 2 * q3q3 + 1)

        # print('A:%.2f H:%.2f V:%.2f R_x:%.2f' %
        #       (self.get_angle(self.v, tv2), self.get_angle(self.v, h_tv2), self.get_angle(self.v, v_tv2), self.angle.x * 57.3), end='\r')
        output_q.put(tv2)

        # 存储更替相应的四元数
        self.his_q0 = q0
        self.his_q1 = q1
        self.his_q2 = q2
        self.his_q3 = q3

    def read_data(self):
        data_frame, c_time = self.input_q.get(True)
        self.delta_time = (c_time - self.pre_time).microseconds / 1e6
        self.pre_time = c_time
        return data_frame

    def data_offset(self, cnt):
        # global ser, wa, c, frame_len, data, pre_time, now_time, delta_time, start
        # global acc_offset_x, acc_offset_y, acc_offset_z, gyro_offset_x, gyro_offset_y, gyro_offset_z
        sum_acc = Acc(0.0, 0.0, 0.0)
        sum_gyro = Gyro(0.0, 0.0, 0.0)
        length = cnt

        while True:
            data_frame = self.read_data()
            t_Acc = Acc()
            t_Gyro = Gyro()
            (t_Acc.x, t_Acc.y, t_Acc.z) = self.get_acc(data_frame)
            (t_Gyro.x, t_Gyro.y, t_Gyro.z, _) = self.get_gyro(data_frame)
            sum_acc.x += t_Acc.x
            sum_acc.y += t_Acc.y
            # sum_acc.z += t_Acc.z
            sum_gyro.x += t_Gyro.x
            sum_gyro.y += t_Gyro.y
            sum_gyro.z += t_Gyro.z
            cnt -= 1
            if cnt % 10 == 0:
                print('offset left:', cnt)
            if cnt == 0:
                self.acc_offset_x = sum_acc.x / length
                self.acc_offset_y = sum_acc.y / length
                self.acc_offset_z = sum_acc.z / length
                self.gyro_offset_x = sum_gyro.x / length
                self.gyro_offset_y = sum_gyro.y / length
                self.gyro_offset_z = sum_gyro.z / length
                print('acc_offset:', self.acc_offset_x, self.acc_offset_y, self.acc_offset_z)
                print('gyro_offset:', self.gyro_offset_x, self.gyro_offset_y, self.gyro_offset_z)
                break


def process(input_q, output_q):
    itg3205 = ITG3205(input_q)
    while True:
        data_frame = itg3205.read_data()
        ax, ay, az = itg3205.acc(itg3205.get_acc(data_frame))
        gx, gy, gz, _ = itg3205.angular_velocity(data_frame)
        itg3205.imu_update(gx, gy, gz, ax, ay, az, output_q)


def collect(output_q1, output_q2):
    while True:
        v1 = output_q1.get(True)
        v2 = output_q2.get(True)
        angle = ITG3205.get_angle(v1, v2)
        print(v1)
        print(v2)
        print(angle)


if __name__ == '__main__':
    input_q1 = Queue()
    output_q1 = Queue()
    input_q2 = Queue()
    output_q2 = Queue()

    p1 = Process(target=process, args=(input_q1, output_q1))
    p2 = Process(target=process, args=(input_q2, output_q2))
    collector = Process(target=collect, args=(output_q1, output_q2))
    p1.start()
    p2.start()
    collector.start()

    print('processes started')

    ser = serial.Serial('COM6', 115200)
    frame_len = 80
    data = [0] * frame_len
    (wa, c) = (0,) * 2

    while True:
        if ser.read() == b'\xaa' and ser.read() == b'\xaa':
            now_time = datetime.datetime.now()
            cou = 0
            c = c + 1
            while True:
                t = ser.read()
                cou = cou + 1
                if t != b'\xee':
                    if 0 <= cou - 1 < frame_len:
                        data[cou - 1] = t
                else:
                    t = ser.read()
                    cou = cou + 1
                    if t == b'\xee':
                        cou = cou - 2
                        break
                    else:
                        if 0 <= cou - 2 < frame_len:
                            data[cou - 2] = b'\xee'
                        if 0 <= cou - 1 < frame_len:
                            data[cou - 1] = t
            if cou != frame_len:
                wa = wa + 1
                print(str(wa) + ' in ' + str(c))
            else:
                input_q1.put((data[40:60], now_time))
                input_q2.put((data[60:80], now_time))
