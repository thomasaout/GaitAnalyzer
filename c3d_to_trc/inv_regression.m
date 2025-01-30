function target = inv_regression(a,b,c,coeff)
% fonction inverse de best regression

	target  = a + coeff(1)*(b-a) + coeff(2)*(c-a) + coeff(3)*cross(b-a,c-a);

end