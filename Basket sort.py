l = [2,3,5,10,2,4,3,8,5]
l2 = [0] * max(l)
l3 = []
for i in l:
    z = i -1
    print(z)
    l2[z] += 1

print(l2)
for i in range(len(l2)):
    anzahl = l2[i]
    for z in range(anzahl):
        l3.append(i+1)
print(l3)