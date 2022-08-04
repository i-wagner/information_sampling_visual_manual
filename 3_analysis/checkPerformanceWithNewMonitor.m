close all;

%%
figure;
tiledlayout(2, 2)

nexttile
line([0, 1], [0, 1]);
hold on;
plot(perf.hitrates(1:90, :, 2), perf.hitrates(1:90, :, 3), 'o');
plot(perf.hitrates(91:end, :, 2), perf.hitrates(91:end, :, 3), '*');
hold off;
axis([0, 1, 0, 1], 'square')
xlabel('Proportion correct [easy]')
ylabel('Proportion correct [difficult]')


%%
nexttile
line([floor(min(sacc.time.mean.planning(:))), ceil(max(sacc.time.mean.planning(:)))], ...
     [floor(min(sacc.time.mean.planning(:))), ceil(max(sacc.time.mean.planning(:)))]);
hold on;
plot(sacc.time.mean.planning(1:90, :, 2),   sacc.time.mean.planning(1:90, :, 3), 'o');
plot(sacc.time.mean.planning(91:end, :, 2), sacc.time.mean.planning(91:end, :, 3), '*');
hold off;
axis([floor(min(sacc.time.mean.planning(:))), ceil(max(sacc.time.mean.planning(:))), ...
      floor(min(sacc.time.mean.planning(:))), ceil(max(sacc.time.mean.planning(:)))], 'square')
xlabel('Planning time [easy]')
ylabel('Planning time [difficult]')


%%
nexttile
line([floor(min(sacc.time.mean.inspection(:))), ceil(max(sacc.time.mean.inspection(:)))], ...
     [floor(min(sacc.time.mean.inspection(:))), ceil(max(sacc.time.mean.inspection(:)))]);
hold on;
plot(sacc.time.mean.inspection(1:90, :, 2),   sacc.time.mean.inspection(1:90, :, 3), 'o');
plot(sacc.time.mean.inspection(91:end, :, 2), sacc.time.mean.inspection(91:end, :, 3), '*');
hold off;
axis([floor(min(sacc.time.mean.inspection(:))), ceil(max(sacc.time.mean.inspection(:))), ...
      floor(min(sacc.time.mean.inspection(:))), ceil(max(sacc.time.mean.inspection(:)))], 'square')
xlabel('Inspection time [easy]')
ylabel('Inspection time [difficult]')


%%
nexttile
line([floor(min(sacc.time.mean.decision(:))), ceil(max(sacc.time.mean.decision(:)))], ...
     [floor(min(sacc.time.mean.decision(:))), ceil(max(sacc.time.mean.decision(:)))]);
hold on;
plot(sacc.time.mean.decision(1:90, :, 2),   sacc.time.mean.decision(1:90, :, 3), 'o');
plot(sacc.time.mean.decision(91:end, :, 2), sacc.time.mean.decision(91:end, :, 3), '*');
hold off;
axis([floor(min(sacc.time.mean.decision(:))), ceil(max(sacc.time.mean.decision(:))), ...
      floor(min(sacc.time.mean.decision(:))), ceil(max(sacc.time.mean.decision(:)))], 'square')
xlabel('Decision time [easy]')
ylabel('Decision time [difficult]')