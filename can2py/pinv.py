#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def main():
    A = np.array([[1.,1.]    # 行列Aの生成
                 ,[1.,1.]])
    pinvA = np.linalg.pinv(A)             # Aの擬似逆行列
    print( "invA=\n" + str(pinvA) )      # 計算結果の表示

if __name__ == '__main__':
    main()
