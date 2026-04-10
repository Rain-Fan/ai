import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 检查环境
print(f"Python 版本: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Matplotlib: {plt.matplotlib.__version__}")



# 设置随机种子以保证结果可复现
np.random.seed(42)

# 模拟生成 200 套房子的数据
n_houses = 200
data = {
    'district': np.random.choice(['海曙', '江北', '鄞州', '镇海', '北仑'], n_houses),
    'area_sqm': np.random.normal(100, 30, n_houses).clip(40, 250), # 面积
    'bedrooms': np.random.randint(1, 5, n_houses),                # 卧室数
    'age_years': np.random.randint(0, 30, n_houses),              # 房龄
    'floor': np.random.randint(1, 30, n_houses)                   # 楼层
}

# 将数据转换为 Pandas DataFrame
df = pd.DataFrame(data)

# 根据面积和随机噪声生成价格 (万元)
# 假设基础单价为 2.5 万/平米
df['price_wan'] = (df['area_sqm'] * 2.5 + np.random.normal(0, 20, n_houses)).round(1)

print("数据集已生成，前 5 行如下：")
print(df.head())



# 1. 查看数据概况
print("\n--- 数据基本信息 ---")
print(df.info())

# 2. 查看统计摘要（均值、标准差等）
print("\n--- 统计摘要 ---")
print(df.describe())

# 3. 创建新列：计算每平米单价
df['price_per_sqm'] = df['price_wan'] / df['area_sqm']

# 4. 分组统计：计算各区的平均房价
print("\n--- 各区平均房价 (万元) ---")
print(df.groupby('district')['price_wan'].mean().sort_values(ascending=False))



# 设置画布大小
plt.figure(figsize=(12, 5))

# 绘图 1：面积与价格的散点图 (寻找相关性)
plt.subplot(1, 2, 1)
plt.scatter(df['area_sqm'], df['price_wan'], alpha=0.6, color='blue')
plt.title('Area vs Price')
plt.xlabel('Area (sqm)')
plt.ylabel('Price (Wan)')
plt.grid(True, alpha=0.3)

# 绘图 2：房价分布直方图 (观察分布形状)
plt.subplot(1, 2, 2)
plt.hist(df['price_wan'], bins=20, color='green', edgecolor='black', alpha=0.7)
plt.title('Distribution of House Prices')
plt.xlabel('Price (Wan)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()