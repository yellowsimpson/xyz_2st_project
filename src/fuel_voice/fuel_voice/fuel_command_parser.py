import re

import rclpy
from rclpy.node import Node

from fuel_interfaces.msg import FuelCommand


class FuelCommandParser(Node):
    def __init__(self):
        super().__init__('fuel_command_parser')

        # /fuel_command 토픽 퍼블리셔
        self.pub = self.create_publisher(FuelCommand, 'fuel_command', 10)

        self.get_logger().info('FuelCommandParser node started.')
        self.get_logger().info('터미널에 "경유 3만원" 같은 문장을 입력하면 됩니다.')
        self.get_logger().info('종료하려면 Ctrl+C')

        # 타이머로 주기적으로 입력 받기 (비동기적으로 돌기 위해)
        self.timer = self.create_timer(0.1, self.read_input_once)
        self._waiting_input = True

    def read_input_once(self):
        if not self._waiting_input:
            return

        try:
            # blocking input을 timer 안에서 바로 쓰면 안 좋긴 한데,
            # 시작 테스트용이라 간단하게 이렇게 구현
            text = input('\n주유 명령 입력 (예: "경유 3만원"): ')
        except EOFError:
            return

        self._waiting_input = False
        if not text.strip():
            self._waiting_input = True
            return

        cmd_msg = self.parse_text_to_command(text)
        if cmd_msg is None:
            self.get_logger().warn('파싱 실패: "%s"' % text)
        else:
            self.pub.publish(cmd_msg)
            self.get_logger().info(f'퍼블리시: {cmd_msg}')

        # 다시 입력 기다리도록 플래그 리셋
        self._waiting_input = True

    def parse_text_to_command(self, text: str):
        """
        "경유 3만원", "휘발유 5리터" 같은 문장을 FuelCommand로 변환
        """
        msg = FuelCommand()
        msg.intent = 'fuel_request'
        msg.raw_text = text

        lower = text.replace(' ', '')  # 공백 제거 (예: "3 만원" -> "3만원")

        # 1) 연료 타입 파싱
        # 기본값
        fuel_type = None

        if '경유' in lower:
            fuel_type = 'diesel'
        elif '휘발유' in lower or '가솔린' in lower:
            fuel_type = 'gasoline'
        # 나중에 LPG, 전기 등도 추가 가능

        if fuel_type is None:
            self.get_logger().warn('연료 타입을 찾지 못했습니다 (경유/휘발유).')
            return None

        msg.fuel_type = fuel_type

        # 2) 금액/리터 파싱
        # 패턴 예:
        # - 3만원, 3만, 30000원, 3만5천원
        # - 5리터, 5L
        amount_type = None
        amount_value = None
        unit = ''

        # (1) 리터 기준인지 먼저 확인
        liter_match = re.search(r'(\d+(\.\d+)?)\s*(리터|L|l)', lower)
        if liter_match:
            amount_type = 'volume'
            amount_value = float(liter_match.group(1))
            unit = 'L'
        else:
            # (2) 돈 기준 (원, 만원, 원 생략 등)
            # 매우 단순 버전: 숫자 + "만원" 또는 "원"
            # 예: 3만원 -> 30000, 30000원 -> 30000
            # "3만5천원" 이런 건 나중 확장
            manwon_match = re.search(r'(\d+)\s*만\s*원?', lower)
            if manwon_match:
                amount_type = 'money'
                amount_value = float(manwon_match.group(1)) * 10000.0
                unit = 'KRW'
            else:
                # 그냥 숫자 + 원
                won_match = re.search(r'(\d+)\s*원', lower)
                if won_match:
                    amount_type = 'money'
                    amount_value = float(won_match.group(1))
                    unit = 'KRW'
                else:
                    # "경유 3만원"처럼 "만"과 "원"이 붙어있을 수도 있으니
                    man_only = re.search(r'(\d+)\s*만', lower)
                    if man_only:
                        amount_type = 'money'
                        amount_value = float(man_only.group(1)) * 10000.0
                        unit = 'KRW'

        if amount_type is None or amount_value is None:
            self.get_logger().warn('금액/리터 정보를 찾지 못했습니다.')
            return None

        msg.amount_type = amount_type
        msg.amount_value = float(amount_value)
        msg.unit = unit

        return msg


def main(args=None):
    rclpy.init(args=args)
    node = FuelCommandParser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
