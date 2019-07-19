import actor_critic as ppo
x = ppo.actor()
y= [0.7]*30
Y = [0.4]*30
Y[1]= -.3
Y[2]= -.3
Y[3]= -.3
Y[0]= -.3
z=[0.5]*70
o= [.1]*70
k= list(-2**(n/5) for n in range(70))
try:
	while True:
		#x.fit(z, y, [[10]])
		x.fit(o, Y, [[10]])
except KeyboardInterrupt:
	print(x.predict(z))
	print(x.predict(o))
	print(x.predict(k))