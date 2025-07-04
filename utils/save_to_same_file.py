import os

import pandas as pd

# 假设这里是存储文件名的列表，你可以替换为实际的文件名列表
model_names = ['iTransformer', 'Pyraformer', 'Mamba']
for site, dataset in zip(['Tibet', 'Tibet', 'USA', 'Australia'], ['2018', '2019', '2019', '2019']):

    # 根据站点和数据集构建文件路径列表
    file_names = [f'../data_record/{site}/{dataset}/Reconcile_Metrics_hierarchy_{name}_{site}_{dataset}.xlsx' for name
                  in model_names]

    new_file_path = '../data_record/reconcile_metrics_all.xlsx'

    # 遍历每个文件
    for file_name in file_names:
        try:
            with pd.ExcelFile(file_name) as xls:
                sheets = xls.sheet_names

                # 遍历文件中的每个sheet
                for sheet in sheets:
                    df = pd.read_excel(xls, sheet_name=sheet)

                    # 获取最后7行，前10列的数据
                    last_rows = df.iloc[-7:, :11].reset_index(drop=True)

                    # 如果文件已存在，打开并附加数据
                    if os.path.exists(new_file_path):
                        with pd.ExcelWriter(new_file_path, engine='openpyxl', mode='a',
                                            if_sheet_exists='overlay') as writer:
                            # 获取当前sheet的最大行数，确保数据写入不会覆盖现有内容
                            if sheet in writer.sheets:
                                startrow = writer.sheets[sheet].max_row
                            else:
                                startrow = 0  # 如果sheet不存在，从第0行开始写入

                            # 将数据写入对应的sheet，避免重复写入header
                            last_rows.to_excel(writer, sheet_name=sheet, startrow=startrow, header=False, index=False)
                    else:
                        # 如果文件不存在，创建并写入数据
                        with pd.ExcelWriter(new_file_path, engine='xlsxwriter') as writer:
                            last_rows.to_excel(writer, sheet_name=sheet, index=False)

                print(f"已成功将 {file_name} 文件的每个sheet的最后7行前11列数据保存至 {new_file_path} 对应的sheet中。")
        except Exception as e:
            print(f"处理文件 {file_name} 出现错误: {e}")
