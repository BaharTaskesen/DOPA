
clear all
close all
clc

load bisection_timer_1e4
load ftrl_timer_1e4w

reps = 10;
K_max = 1e4;

ind = (1:9)' * logspace(0, log10(K_max)-1, log10(K_max)); 
ind = [ind(:)', K_max];
colors = [108, 111, 116; 172, 80, 137;] ./ 255;
fig = figure;
hold on
p_ftrl = plot_with_shade(ind, ftrl_timer', 5, 0.2, colors(1,:));
p_bisection = plot_with_shade(ind, bisection_timer', 5, 0.2, colors(2,:));

set(gca, 'XScale', 'log', 'YScale', 'log');
set(gca, 'FontSize', 16);
xlabel('$\#$ of arms ($K$)', 'FontSize', 20, 'interpreter','latex');
ylabel('Execution time (s)','FontSize', 20, 'interpreter','latex')
grid on
lgd = legend([p_bisection, p_ftrl], 'DOPA', 'FTRL',...
    'Location', 'southwest', 'interpreter','latex');
lgd.FontSize = 16;
remove_border()

saveas(gcf, 'execution_time', 'svg')
saveas(gcf, 'execution_time', 'pdf')