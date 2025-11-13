import sys
print(f"Python版本: {sys.version}")
assert sys.version_info >= (3, 8), "需要Python 3.8或更高版本！"

# 检查关键库
import pandas as pd
import sklearn
print(f"pandas版本: {pd.__version__}")
print(f"sklearn版本: {sklearn.__version__}")