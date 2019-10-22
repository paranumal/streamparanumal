
GimbutasXiaoCubatureTet3D

V = 1/3*(1/2)*(2*2)*2

sc = (4/3)/sqrt(8/9);

maxNcub = length(cubXYZW);

fp = fopen('GimbutasXiaoCubatureTet3D.c','w');

fprintf(fp, '#include <stdio.h>\n');
fprintf(fp, '#include <stdlib.h>\n\n');

fprintf(fp, 'int cubNps[%d] = {', maxNcub);
for Ncub=1:maxNcub-1
  fprintf(fp, '%d, ',  length(cubXYZW{Ncub}));
end
fprintf(fp, '%d};\n\n',  length(cubXYZW{end}));

for Ncub=1:maxNcub
  
  x = cubXYZW{Ncub}(:,1);
  y = cubXYZW{Ncub}(:,2);
  z = cubXYZW{Ncub}(:,3);
  w = cubXYZW{Ncub}(:,4);

  [cubr,cubs,cubt] = xyztorst(x,y,z);
  [min(cubr),max(cubr)]
  [min(cubs),max(cubs)]
  [min(cubt),max(cubt)]

  cubw = w*sc;

  fprintf(fp, 'dfloat cubR%d[%d] = {', Ncub, length(cubr));
  for c=1:length(cubr)-1
    fprintf(fp, '% 17.15e,', cubr(c));
  end
  fprintf(fp, '% 17.15e};\n', cubr(end));

  fprintf(fp, 'dfloat cubS%d[%d] = {', Ncub, length(cubs));
  for c=1:length(cubs)-1
    fprintf(fp, '% 17.15e,', cubs(c));
  end
  fprintf(fp, '% 17.15e};\n', cubs(end));

  fprintf(fp, 'dfloat cubT%d[%d] = {', Ncub, length(cubt));
  for c=1:length(cubt)-1
    fprintf(fp, '% 17.15e,', cubt(c));
  end
  fprintf(fp, '% 17.15e};\n', cubt(end));

  fprintf(fp, 'dfloat cubW%d[%d] = {', Ncub, length(cubw));
  for c=1:length(cubw)-1
    fprintf(fp, '% 17.15e,', cubw(c));
  end
  fprintf(fp, '% 17.15e};\n\n', cubw(end));
  
end

fprintf(fp, 'int GimbutasXiaoCubatureTet3D(int cubN, dfloat **cubr, dfloat **cubs, dfloat **cubt, dfloat **cubw){\n');

fprintf(fp, '  int cubNp = cubNps[cubN];\n\n');
fprintf(fp, '  *cubr = (dfloat*) calloc(cubNp, sizeof(dfloat)); \n');
fprintf(fp, '  *cubs = (dfloat*) calloc(cubNp, sizeof(dfloat)); \n');
fprintf(fp, '  *cubt = (dfloat*) calloc(cubNp, sizeof(dfloat)); \n');
fprintf(fp, '  *cubw = (dfloat*) calloc(cubNp, sizeof(dfloat)); \n\n');

fprintf(fp, '  dfloat *cubR, *cubS, *cubT, *cubW;\n');
fprintf(fp, '  switch(cubN){\n');
for n=1:maxNcub
  fprintf(fp, '    case %d: cubR = cubR%d; cubS = cubS%d; cubT = cubT%d; cubW = cubW%d; break;\n',n,n,n,n,n);
end
fprintf(fp, '    default: exit(-1); \n');
fprintf(fp, '  }\n\n');

fprintf(fp, '  for(int n=0;n<cubNp;++n){\n');
fprintf(fp, '    cubr[0][n] = cubR[n];\n');
fprintf(fp, '    cubs[0][n] = cubS[n];\n');
fprintf(fp, '    cubt[0][n] = cubT[n];\n');
fprintf(fp, '    cubw[0][n] = cubW[n];\n');
fprintf(fp, '  }\n\n');

fprintf(fp, ' return cubNp;\n');
fprintf(fp, '}\n\n');


fclose(fp);
