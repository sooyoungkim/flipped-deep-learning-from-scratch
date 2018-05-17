from ch05 import layer_naive as na

apple = 100
apple_num = 2
tax = 1.1

# 계층들
mul_apple_layer = na.MulLayer()
mul_tax_layer = na.MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)    # 220.00000000000003
print("price:", int(price))

# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)     # 2.2 110.00000000000001 200
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))
print("dTax:", dtax)