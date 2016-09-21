import subprocess
import numpy

loop0avg = []
loop1avg = []
loop2avg = []
loop3avg = []
gflopsavg = []
gbsavg = []
answeravg = []

start = 997
end = 1004
for i in range (start, end, 1):
	for j in range (start, end, 1):
		loop0 = []
		loop1 = []
		loop2 = []
		loop3 = []
		gflops = []
		gbs = []
		answer = []
		for k in range (0,3):
			print("executing test: " + str(i) + " " + str(j) + ", iteration " + str(k))
			cmd = "./cs257 " + str(i) + " " + str(j) + " 0"
			p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
			output = p.stdout.read().strip().split('\n')
			for line in output:
				if 'Loop 0' in line:
					number = line.split(' ')[4]
					loop0.append(float(number))
				if 'Loop 1' in line:
					number = line.split(' ')[4]
					loop1.append(float(number))
				if 'Loop 2' in line:
					number = line.split(' ')[4]
					loop2.append(float(number))
				if 'Loop 3' in line:
					number = line.split(' ')[4]
					loop3.append(float(number))
				if 'GFLOP' in line:
					number = line.split(' ')[3]
					gflops.append(float(number))
				if 'GB/s' in line:
					number = line.split(' ')[3]
					gbs.append(float(number))
				if 'Answer' in line:
					number = line.split(' ')[3]
					answer.append(float(number))
		loop0avg.append(sum(loop0)/3)
		loop1avg.append(sum(loop1)/3)
		loop2avg.append(sum(loop2)/3)
		loop3avg.append(sum(loop3)/3)
		gflopsavg.append(sum(gflops)/3)
		gbsavg.append(sum(gbs)/3)
		answeravg.append(sum(answer)/3)


length = len(loop0avg);

loop0 = sum(loop0avg)/length
loop1 = sum(loop1avg)/length
loop2 = sum(loop2avg)/length
loop3 = sum(loop3avg)/length
flops = sum(gflopsavg)/length
gbs = sum(gbsavg)/length

fo = open("results.txt", "w")
fo.write(str(loop0avg)+"\n")
fo.write(str(loop1avg)+"\n")
fo.write(str(loop2avg)+"\n")
fo.write(str(loop3avg)+"\n")
fo.write(str(gflopsavg)+"\n")
fo.write(str(gbsavg)+"\n")
fo.write(str(answeravg)+"\n")
fo.write(str(loop0)+"\n")
fo.write(str(loop1)+"\n")
fo.write(str(loop2)+"\n")
fo.write(str(loop3)+"\n")
fo.write(str(flops)+"\n")
fo.write(str(gbs)+"\n")
fo.close()
