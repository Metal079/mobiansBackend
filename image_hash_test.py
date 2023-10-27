from PIL import Image
import imagehash
hash0 = imagehash.crop_resistant_hash(Image.open('tes1.png')) 
hash1 = imagehash.crop_resistant_hash(Image.open('tes2.png')) 
cutoff = 5  # maximum bits that could be different between the hashes. 

print(hash0)
print(hash0 -  hash1)
if hash0 - hash1 < cutoff:
  print('images are similar')
else:
  print('images are not similar')