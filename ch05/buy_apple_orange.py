from ch05 import layer_naive_mul_add as na

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 계층들
mul_apple_layer = na.MulLayer()
mul_orange_layer = na.MulLayer()
add_apple_orange_layer = na.AddLayer()
mul_tax_layer = na.MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)  # (1)
orange_price = mul_orange_layer.forward(orange, orange_num)  # (2)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)  # (3)
price = mul_tax_layer.forward(all_price, tax)  # (4)
print("price:", int(price))     # price: 715

# 역전파
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print("dApple:", dapple)                # dApple: 2.2
print("dApple_num:", int(dapple_num))   # dApple_num: 110
print("dOrange:", dorange)              # dOrange: 3.3000000000000003
print("dOrange_num:", int(dorange_num)) # dOrange_num: 165
print("dTax:", dtax)                    # dTax: 650

