import pandas as pd

input_path = "/home/liuyou/Downloads/egol1.csv"
output_path = "/home/liuyou/Downloads/egol1_deduplicated.csv"

# 按字符串读取，保留编号的前导零和原始文本
df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
result = df.drop_duplicates(keep="first")

result.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"原始行数：{len(df)}")
print(f"去重后行数：{len(result)}")
print(f"删除重复行：{len(df) - len(result)}")
print(f"输出文件：{output_path}")