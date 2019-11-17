function generateFigure(imgW, imgH)

figure;
x = [0: 0.1 : 2*pi];
y = x.*x;
plot(x, cos(x), "-g;cos(x);", x, sin(x), "-r;sin(x);", x, y , "-b;x^2;");
xlabel('x');
ylabel('y');
axis([0 2*pi -1 1]);

% ============================================================

end
