
GimbutasXiaoCubatureTri2D

sc = 2/sqrt(3);

maxNcub = length(cubXYW);

fp = fopen('GimbutasXiaoCubatureTri2D.cpp','w');

fprintf(fp, '#include <stdio.h>\n');
fprintf(fp, '#include <stdlib.h>\n\n');

fprintf(fp, 'int cubNps[%d] = {', maxNcub);
for Ncub=1:maxNcub-1
  fprintf(fp, '%d, ',  length(cubXYW{Ncub}));
end
fprintf(fp, '%d};\n\n',  length(cubXYW{end}));

for Ncub=1:maxNcub
  
  x = cubXYW{Ncub}(:,1);
  y = cubXYW{Ncub}(:,2);
  cubw = sc*cubXYW{Ncub}(:,3);
sum(cubw)
  [cubr,cubs] = xytors(x,y);
[min(cubr),max(cubr)];
[min(cubs),max(cubs)];
[min(1+cubr+cubs),max(1+cubr+cubs)];

  fprintf(fp, 'dfloat cubTriR%d[%d] = {', Ncub, length(cubr));
  for c=1:length(cubr)-1
    fprintf(fp, '% 17.15e,', cubr(c));
  end
  fprintf(fp, '% 17.15e};\n', cubr(end));

  fprintf(fp, 'dfloat cubTriS%d[%d] = {', Ncub, length(cubs));
  for c=1:length(cubs)-1
    fprintf(fp, '% 17.15e,', cubs(c));
  end
  fprintf(fp, '% 17.15e};\n', cubs(end));

  fprintf(fp, 'dfloat cubTriW%d[%d] = {', Ncub, length(cubw));
  for c=1:length(cubw)-1
    fprintf(fp, '% 17.15e,', cubw(c));
  end
  fprintf(fp, '% 17.15e};\n\n', cubw(end));
  
end

fprintf(fp, 'int GimbutasXiaoCubatureTri2D(int cubN, dfloat **cubr, dfloat **cubs, dfloat **cubt, dfloat **cubw){\n');

fprintf(fp, '  int cubNp = cubNps[cubN-1];\n\n');
fprintf(fp, '  *cubr = (dfloat*) calloc(cubNp, sizeof(dfloat)); \n');
fprintf(fp, '  *cubs = (dfloat*) calloc(cubNp, sizeof(dfloat)); \n');
fprintf(fp, '  *cubw = (dfloat*) calloc(cubNp, sizeof(dfloat)); \n\n');

fprintf(fp, '  dfloat *cubR, *cubS, *cubW;\n');
fprintf(fp, '  switch(cubN){\n');
for n=1:maxNcub
  fprintf(fp, '    case %d: cubR = cubTriR%d; cubS = cubTriS%d; cubW = cubTriW%d; break;\n',n,n,n,n);
end
fprintf(fp, '    default: exit(-1); \n');
fprintf(fp, '  }\n\n');

fprintf(fp, '  for(int n=0;n<cubNp;++n){\n');
fprintf(fp, '    cubr[0][n] = cubR[n];\n');
fprintf(fp, '    cubs[0][n] = cubS[n];\n');
fprintf(fp, '    cubw[0][n] = cubW[n];\n');
fprintf(fp, '  }\n\n');

fprintf(fp, ' return cubNp;\n');
fprintf(fp, '}\n\n');


fclose(fp);
