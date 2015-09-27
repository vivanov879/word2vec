require 'nn'


x = torch.rand(2,3)
m = nn.Normalize(2)
print(m.p)
o = m:forward(x)

print(x)
print(o:size())
print(o)

