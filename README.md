# AI-D-project
AI-D project for power converter using BN-NN and GA

## 開發工具
IDE: VSCODE

環境:w11/python 3.12

## 環境建置
1. 安裝vscode並重新啟動(記得安裝python extension)

2. 建立資料夾a.虛擬環境資料夾b.程式碼資料夾

3. 把電路檔案以及程式碼移至b.資料夾

4. 在a.資料夾建立虛擬環境
```
python -m venv ltspiceauto-venv
```
5. 檢查虛擬環境是否建立成功
```
cd desktop\mywork
ltspiceauto-venv\scripts\activate
cd ltspiceauto-code
```
6. 檢查Python版本以及安裝pip工具
```
python --version
pip list
```
7. 在vscode加入終端機方便後續使用，
在vscode使用快捷鍵ctrl+shift+p
搜尋View: Toggle Terminal

9. 設定終端機可以執行python venv指令
這時請先以系統管理員的身分打開 PowerShell，並輸入 get-executionpolicy 就會發現現在是處於 Restricted 的狀態。
輸入 set-executionpolicy RemoteSigned，並且選擇 A 就能夠覆寫初始設定

10. 測試vscode內建終端機能否啟動虛擬資料夾
ltspiceauto-venv\scripts\activate

11. 安裝pyltspice工具/PyMySQL工具,分別為ltspice自動化模組以及資料庫連結模組
```
pip install PyLTSpice
pip install --upgrade PyLTSpice
pip install PyMySQL
```
11. 安裝SQL server 以連線至資料庫
本機資料庫需新增使用者/密碼 (server-> Users and privileges)

## 執行主程式前注意事項
1. 模擬資料筆數設定,例如20就是開關頻率、電感值、電容值分20等份共8000筆資料
```
ltspiceautorun_v_net.py
```
```python
datas = sc.training_design_parameter(20)
```
2. 設定參數範圍
```
select_com_bn_nn.py
```
```python
f_range = {'f_min':20e3,'f_max':200e3} #Hz
c_range = {'c_min':47e-6,'c_max':726e-6} #F
l_range = {'l_min':100e-6,'l_max':1000e-6} #H
```
3. 設定lookup table 範圍
```
gen_lookup_table.py
```
```python
f_range = {'f_min':20e3,'f_max':200e3} #Hz
c_range = {'c_min':47,'c_max':726} #uF
l_range = {'l_min':100,'l_max':1000} #uH
```
4. 輸入電路檔案位置.asc檔以及模擬結果儲存位置(例如存在temp資料夾)
```
ltspiceautorun_v_net.py
```
```python
LTC = SimRunner(parallel_sims=1, output_folder='./temp')
LTC.create_netlist('./circuitfile/synchronous_buck_va_v3.asc')
netlist = SpiceEditor('./circuitfile/synchronous_buck_va_v3.net')
```
## 執行流程
1. 到虛擬環境資料夾
```
cd desktop\mywork
```
2. 執行虛擬環境
```
ltspiceauto-venv\scripts\activate
```
3. 到主程式資料夾
```
cd ltspiceauto-code
```
4. 產生lookup table csv 檔
```
python write_lookuptable_csv.py
```
5. 執行主程式
```
python -X utf8 ltspiceautorun_v_net.py
```
6. 把主程式產生的訓練資料以及lookup table csv 上傳到google colab 資料夾
```
在google colab 上創建名為data資料夾,把上述csv檔放入,訓練資料需手動刪除第一列名稱
```
7. 在google colab 上運作BN-NN.ipynb檔執行神經網路訓練以及基因演算法找出電路最佳參數

p.s在本機運行tensorflow以及pygad需額外安裝模組，以及在google colab上將.ipynb檔轉為.py檔
```
pip install tensorflow
pip install pygad
pip install -U scikit-learn
```

