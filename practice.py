a = ['200_epoch.bin', '100_epoch.bin', '300_epoch.bin', 'prompt.bin']

# 끝이 bin으로 끝나는 파일만 정렬
print(sorted([i for i in a if i.endswith('bin')]))
