max_nmi_value = float('-inf')  # 初始化为负无穷大

with open('output5_temp.txt', 'r') as file:
    for line in file:
        if "nmi_value" in line:
            nmi_value_str = line.split("nmi_value:")[1].split(",")[0]
            nmi_value = float(nmi_value_str)
            if nmi_value > max_nmi_value:
                max_nmi_value = nmi_value

print("最大的nmi_value是:", max_nmi_value)