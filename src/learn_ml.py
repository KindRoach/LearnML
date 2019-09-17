import numpy

random_array = numpy.random.rand(4, 4)
# print(random_array)

random_mat = numpy.mat(random_array)
# print(random_mat)

invert_mat = random_mat.I
print(invert_mat * random_mat - numpy.eye(4))
