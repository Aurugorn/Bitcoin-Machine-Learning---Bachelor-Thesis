import auger
from BitcoinComponent import BitcoinTransformer

def main():
  p = BitcoinTransformer()
  print(p)

with auger.magic([BitcoinTransformer]):
    main()