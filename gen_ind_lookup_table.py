import pymysql
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(verbose=True)
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

#資料庫參數設定
load_dotenv()
db_settings_core = {
    "host": DB_HOST,
    "port": int(DB_PORT),
    "user": DB_USER,
    "password": DB_PASSWORD,
    "db": "core_database",
    "charset": "utf8"
}

try:
    #建立connection物件

    conn_core= pymysql.connect(**db_settings_core)

    #建立cursor物件

    with conn_core.cursor() as cursor_core:
        # 查詢資料SQL語法
        command = "SELECT * FROM core"

        #執行指令
        cursor_core.execute(command)

        #取得所有資料
        result_core = cursor_core.fetchall()
        print(result_core)

except Exception as ex:
    print(ex)

#從資料庫取得電感參數
#創建電感lookup table (電感值/RL)