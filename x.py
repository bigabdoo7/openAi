import ppo 

x = ppo.actor(0.1)
y= [0.7]*30
z=[.5]*70
x.fit(z, y)
x.fit(z, y)
x.fit(z, y)
x.fit(z, y)
x.fit(z, y)
x.fit(z, y)
x.fit(z, y)
x.fit(z, y)
x.fit(z, y)
print(x.predict(z))