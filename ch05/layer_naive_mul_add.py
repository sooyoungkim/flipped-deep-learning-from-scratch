##############################################################################
# 5.4.1 단순한 계층 구현하기
# (1) 곱셈계층
# (2) 덧셈계층
##############################################################################


# 곱셈계층
class MulLayer:
    def __init__(self):
        # 역전파에서 사용되므로 입력값 x,y를 저장해둔다.
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        # x와 y를 바꾸어 곱한다.
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


# 덧셈계층
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        # dout 값을 그대로 흘려보낸다.
        dx = dout * 1
        dy = dout * 1

        return dx, dy
