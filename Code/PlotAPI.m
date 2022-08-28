clear all

% load('API_NRT_4.mat')
% figure
% plot(API_NRT_4(:,1),'-r')
% hold all
% plot(API_NRT_4(:,2),'--k')
% xlabel('Time (s)')
% ylabel('API near-real time')
% legend('Predicted','Database')
% grid
% 
% load('API_30s_4.mat')
% API_30s_4 = [[0 0]; API_30s_4];
% figure
% plot(API_30s_4(:,1),'-r')
% hold all
% plot(API_30s_4(:,2),'--k')
% f=fit((2:1:length(API_30s_4(:,1)))',API_30s_4(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_30s_4(:,2)))',API_30s_4(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (30 s)')
% ylabel('API 30 s')
% xlim([0 960])
% xtickangle(45);
% grid
% 
% load('API_60s_4.mat')
% API_60s_4 = [[0 0]; API_60s_4];
% figure
% plot(API_60s_4(:,1),'-r')
% hold all
% plot(API_60s_4(:,2),'--k')
% f=fit((2:1:length(API_60s_4(:,1)))',API_60s_4(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_60s_4(:,2)))',API_60s_4(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (60 s)')
% ylabel('API 60 s')
% xlim([0 480])
% xtickangle(45);
% grid
% 
% load('API_30m_4.mat')
% figure
% API_30m_4 = [[0 0]; API_30m_4];
% plot(API_30m_4(:,1),'-r')
% hold all
% plot(API_30m_4(:,2),'--k')
% f=fit((2:1:length(API_30m_4(:,1)))',API_30m_4(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_30m_4(:,2)))',API_30m_4(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (30 m)')
% ylabel('API 30 min')
% grid
% 
% load('API_60m_4.mat')
% API_60m_4 = [[0 0]; API_60m_4];
% figure
% plot(API_60m_4(:,1),'-r')
% hold all
% plot(API_60m_4(:,2),'--k')
% f=fit((2:1:length(API_60m_4(:,1)))',API_60m_4(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_60m_4(:,2)))',API_60m_4(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (60 m)')
% ylabel('API 60 min')
% grid
% 
% 
% load('API_NRT_16.mat')
% figure
% plot(API_NRT_16(:,1),'-r')
% hold all
% plot(API_NRT_16(:,2),'--k')
% xlabel('Time (s)')
% ylabel('API near-real time')
% legend('Predicted','Database')
% grid
% 
% load('API_30s_16.mat')
% API_30s_16 = [[0 0]; API_30s_16];
% figure
% plot(API_30s_16(:,1),'-r')
% hold all
% plot(API_30s_16(:,2),'--k')
% f=fit((2:1:length(API_30s_16(:,1)))',API_30s_16(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_30s_16(:,2)))',API_30s_16(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (30 s)')
% ylabel('API 30 s')
% xlim([0 960])
% grid
% 
% load('API_60s_16.mat')
% API_60s_16 = [[0 0]; API_60s_16];
% figure
% plot(API_60s_16(:,1),'-r')
% hold all
% plot(API_60s_16(:,2),'--k')
% f=fit((2:1:length(API_60s_16(:,1)))',API_60s_16(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_60s_16(:,2)))',API_60s_16(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (60 s)')
% ylabel('API 60 s')
% xlim([0 480])
% grid
% 
% load('API_30m_16.mat')
% figure
% API_30m_16 = [[0 0]; API_30m_16];
% plot(API_30m_16(:,1),'-r')
% hold all
% plot(API_30m_16(:,2),'--k')
% f=fit((2:1:length(API_30m_16(:,1)))',API_30m_16(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_30m_16(:,2)))',API_30m_16(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (30 m)')
% ylabel('API 30 min')
% grid
% 
% load('API_60m_16.mat')
% API_60m_16 = [[0 0]; API_60m_16];
% figure
% plot(API_60m_16(:,1),'-r')
% hold all
% plot(API_60m_16(:,2),'--k')
% f=fit((2:1:length(API_60m_16(:,1)))',API_60m_16(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_60m_16(:,2)))',API_60m_16(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (60 m)')
% ylabel('API 60 min')
% grid


% % % % load('API_30s_4.mat')
% % % % figure
% % % % x=1:1:length(API_30s_4(:,1));
% % % % xx=1:0.1:length(API_30s_4(:,1));
% % % % plot(spline(x,API_30s_4(:,1),xx),'-r')
% % % % hold all
% % % % plot(spline(x,API_30s_4(:,2),xx),'--k')
% % % % xlabel('Time (30 s)')
% % % % ylabel('API 30 s')
% % % % legend('Predicted','Database')
% % % % grid



% load('API_30s_4.mat')
% API_30s_4 = [[0 0]; API_30s_4];
% figure
% plot(smoothdata(movmean(API_30s_4(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_30s_4(:,2),5)),'--k')
% f=fit((1:1:length(API_30s_4(:,1)))',smoothdata(movmean(API_30s_4(:,1),5)),'poly1');
% plot(f,'-.r')
% f=fit((1:1:length(API_30s_4(:,2)))',smoothdata(movmean(API_30s_4(:,2),5)),'poly1');
% plot(f,':k')
% grid
% xlim([0 960])
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (30 s)')
% ylabel('API 30 s')
%
% load('API_60s_4.mat')
% API_60s_4 = [[0 0]; API_60s_4];
% figure
% plot(smoothdata(movmean(API_60s_4(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_60s_4(:,2),5)),'--k')
% f=fit((1:1:length(API_60s_4(:,1)))',smoothdata(movmean(API_60s_4(:,1),5)),'poly1');
% plot(f,'-.r')
% f=fit((1:1:length(API_60s_4(:,2)))',smoothdata(movmean(API_60s_4(:,2),5)),'poly1');
% plot(f,':k')
% grid
% xlim([0 480])
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (60 s)')
% ylabel('API 60 s')
% 
% load('API_30m_4.mat')
% figure
% plot(smoothdata(movmean(API_30m_4(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_30m_4(:,2),5)),'--k')
% f=fit((1:1:length(API_30m_4(:,1)))',smoothdata(movmean(API_30m_4(:,1),5)),'poly1');
% a=zeros(16,1);
% for k = 1:16
%     a(k) = f(k);
% end
% plot(a,'-.r')
% f=fit((1:1:length(API_30m_4(:,2)))',smoothdata(movmean(API_30m_4(:,2),5)),'poly1');
% a=zeros(16,1);
% for k = 1:16
%     a(k) = f(k);
% end
% plot(a,':k')
% grid
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (30 min)')
% ylabel('API 30 min')
% 
% load('API_60m_4.mat')
% figure
% plot(smoothdata(movmean(API_60m_4(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_60m_4(:,2),5)),'--k')
% f=fit((1:1:length(API_60m_4(:,1)))',smoothdata(movmean(API_60m_4(:,1),5)),'poly1');
% plot(f,'-.r')
% f=fit((1:1:length(API_60m_4(:,2)))',smoothdata(movmean(API_60m_4(:,2),5)),'poly1');
% plot(f,':k')
% grid
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (60 min)')
% ylabel('API 60 min')


% load('API_30s_16.mat')
% API_30s_16 = [[0 0]; API_30s_16];
% figure
% plot(smoothdata(movmean(API_30s_16(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_30s_16(:,2),5)),'--k')
% f=fit((1:1:length(API_30s_16(:,1)))',smoothdata(movmean(API_30s_16(:,1),5)),'poly1');
% plot(f,'-.r')
% f=fit((1:1:length(API_30s_16(:,2)))',smoothdata(movmean(API_30s_16(:,2),5)),'poly1');
% plot(f,':k')
% grid
% xlim([0 960])
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (30 s)')
% ylabel('API 30 s')
%
% load('API_60s_16.mat')
% API_60s_16 = [[0 0]; API_60s_16];
% figure
% plot(smoothdata(movmean(API_60s_16(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_60s_16(:,2),5)),'--k')
% f=fit((1:1:length(API_60s_16(:,1)))',smoothdata(movmean(API_60s_16(:,1),5)),'poly1');
% plot(f,'-.r')
% f=fit((1:1:length(API_60s_16(:,2)))',smoothdata(movmean(API_60s_16(:,2),5)),'poly1');
% plot(f,':k')
% grid
% xlim([0 480])
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (60 s)')
% ylabel('API 60 s')
% 
% load('API_30m_16.mat')
% figure
% plot(smoothdata(movmean(API_30m_16(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_30m_16(:,2),5)),'--k')
% f=fit((1:1:length(API_30m_16(:,1)))',smoothdata(movmean(API_30m_16(:,1),5)),'poly1');
% a=zeros(16,1);
% for k = 1:16
%     a(k) = f(k);
% end
% plot(a,'-.r')
% f=fit((1:1:length(API_30m_16(:,2)))',smoothdata(movmean(API_30m_16(:,2),5)),'poly1');
% a=zeros(16,1);
% for k = 1:16
%     a(k) = f(k);
% end
% plot(a,':k')
% grid
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (30 min)')
% ylabel('API 30 min')
% 
% load('API_60m_16.mat')
% figure
% plot(smoothdata(movmean(API_60m_16(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_60m_16(:,2),5)),'--k')
% f=fit((1:1:length(API_60m_16(:,1)))',smoothdata(movmean(API_60m_16(:,1),5)),'poly1');
% plot(f,'-.r')
% f=fit((1:1:length(API_60m_16(:,2)))',smoothdata(movmean(API_60m_16(:,2),5)),'poly1');
% plot(f,':k')
% grid
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (60 min)')
% ylabel('API 60 min')


















% load('API_30s_4.mat')
% figure
% plot(smoothdata(movmean(API_30s_4(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_30s_4(:,2),5)),'--k')
% f=fit((1:1:length(API_30s_4(:,1)))',smoothdata(movmean(API_30s_4(:,1),5)),'poly1');
% plot(f,'-.r')
% f=fit((1:1:length(API_30s_4(:,2)))',smoothdata(movmean(API_30s_4(:,2),5)),'poly1');
% plot(f,':k')
% grid
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (30 s)')
% ylabel('API 30 s')
% 
% load('API_30s_16.mat')
% figure
% plot(smoothdata(movmean(API_30s_16(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_30s_16(:,2),5)),'--k')
% f=fit((1:1:length(API_30s_16(:,1)))',smoothdata(movmean(API_30s_16(:,1),5)),'poly1');
% plot(f,'-.r')
% f=fit((1:1:length(API_30s_16(:,2)))',smoothdata(movmean(API_30s_16(:,2),5)),'poly1');
% plot(f,':k')
% grid
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (30 s)')
% ylabel('API 30 s')





%%%%%% ins

% load('API_30s_17_ins.mat')
% API_30s_17_ins = [[0 0]; API_30s_17_ins];
% figure
% plot(API_30s_17_ins(:,1),'-r')
% hold all
% plot(API_30s_17_ins(:,2),'--k')
% f=fit((2:1:length(API_30s_17_ins(:,1)))',API_30s_17_ins(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_30s_17_ins(:,2)))',API_30s_17_ins(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% ylabel('API 30 s')
% xlim([0 1260])
% xtickangle(45);
% grid

% load('API_60s_17_ins.mat')
% API_60s_17_ins = [[0 0]; API_60s_17_ins];
% figure
% plot(API_60s_17_ins(:,1),'-r')
% hold all
% plot(API_60s_17_ins(:,2),'--k')
% f=fit((2:1:length(API_60s_17_ins(:,1)))',API_60s_17_ins(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_60s_17_ins(:,2)))',API_60s_17_ins(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% ylabel('API 60 s')
% xlim([0 630])
% xtickangle(45);
% grid

% load('API_30m_17_ins.mat')
% figure
% API_30m_17_ins = [[0 0]; API_30m_17_ins];
% plot(API_30m_17_ins(:,1),'-r')
% hold all
% plot(API_30m_17_ins(:,2),'--k')
% f=fit((2:1:length(API_30m_17_ins(:,1)))',API_30m_17_ins(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_30m_17_ins(:,2)))',API_30m_17_ins(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% ylabel('API 30 min')
% xtickangle(45);
% grid

% load('API_60m_17_ins.mat')
% figure
% API_60m_17_ins = [[0 0]; API_60m_17_ins];
% plot(API_60m_17_ins(:,1),'-r')
% hold all
% plot(API_60m_17_ins(:,2),'--k')
% f=fit((2:1:length(API_60m_17_ins(:,1)))',API_60m_17_ins(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_60m_17_ins(:,2)))',API_60m_17_ins(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% ylabel('API 60 min')
% xtickangle(45);
% grid


% load('API_30s_17_ins.mat')
% API_30s_17_ins = [[0 0]; API_30s_17_ins];
% figure
% plot(smoothdata(movmean(API_30s_17_ins(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_30s_17_ins(:,2),5)),'--k')
% f=fit((1:1:length(API_30s_17_ins(:,1)))',smoothdata(movmean(API_30s_17_ins(:,1),5)),'poly1');
% plot(f,'-.r')
% f=fit((1:1:length(API_30s_17_ins(:,2)))',smoothdata(movmean(API_30s_17_ins(:,2),5)),'poly1');
% plot(f,':k')
% grid
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% xlim([0 960])
% ylabel('API 30 s')
% xtickangle(45);

% load('API_60s_17_ins.mat')
% API_60s_17_ins = [[0 0]; API_60s_17_ins];
% figure
% plot(smoothdata(movmean(API_60s_17_ins(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_60s_17_ins(:,2),5)),'--k')
% f=fit((1:1:length(API_60s_17_ins(:,1)))',smoothdata(movmean(API_60s_17_ins(:,1),5)),'poly1');
% plot(f,'-.r')
% f=fit((1:1:length(API_60s_17_ins(:,2)))',smoothdata(movmean(API_60s_17_ins(:,2),5)),'poly1');
% plot(f,':k')
% grid
% xlim([0 480])
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% ylabel('API 60 s')
% xtickangle(45);


% load('API_30m_17_ins.mat')
% figure
% plot(smoothdata(movmean(API_30m_17_ins(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_30m_17_ins(:,2),5)),'--k')
% f=fit((1:1:length(API_30m_17_ins(:,1)))',smoothdata(movmean(API_30m_17_ins(:,1),5)),'poly1');
% a=zeros(21,1);
% for k = 1:21
%     a(k) = f(k);
% end
% plot(a,'-.r')
% f=fit((1:1:length(API_30m_17_ins(:,2)))',smoothdata(movmean(API_30m_17_ins(:,2),5)),'poly1');
% a=zeros(21,1);
% for k = 1:21
%     a(k) = f(k);
% end
% plot(a,':k')
% grid
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% ylabel('API 30 min')
% xtickangle(45);


% load('API_60m_17_ins.mat')
% figure
% plot(smoothdata(movmean(API_60m_17_ins(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_60m_17_ins(:,2),5)),'--k')
% f=fit((1:1:length(API_60m_17_ins(:,1)))',smoothdata(movmean(API_60m_17_ins(:,1),5)),'poly1');
% plot(f,'-.r')
% f=fit((1:1:length(API_60m_17_ins(:,2)))',smoothdata(movmean(API_60m_17_ins(:,2),5)),'poly1');
% plot(f,':k')
% grid
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% ylabel('API 60 min')
% xtickangle(45);


% % % % load('API_30m_17_ins.mat')
% % % % figure
% % % % a=zeros(21,1);
% % % % b=smoothdata(movmean(API_30m_17_ins(:,1),5));
% % % % for k = 2:21
% % % %     a(k) = b(k-1);
% % % % end
% % % % plot(a,'-r')
% % % % hold all
% % % % a=zeros(21,1);
% % % % b=smoothdata(movmean(API_30m_17_ins(:,2),5));
% % % % for k = 2:21
% % % %     a(k) = b(k-1);
% % % % end
% % % % plot(a,'--k')
% % % % a=zeros(21,1);
% % % % b=smoothdata(movmean(API_30m_17_ins(:,1),5));
% % % % for k = 2:21
% % % %     a(k) = b(k-1);
% % % % end
% % % % f=fit((1:1:length(a))',smoothdata(movmean(a,5)),'poly1');
% % % % a=zeros(21,1);
% % % % for k = 1:21
% % % %     a(k) = f(k);
% % % % end
% % % % plot(a,'-.r')
% % % % a=zeros(21,1);
% % % % b=smoothdata(movmean(API_30m_17_ins(:,2),5));
% % % % for k = 2:21
% % % %     a(k) = b(k-1);
% % % % end
% % % % f=fit((1:1:length(a))',smoothdata(movmean(a,5)),'poly1');
% % % % a=zeros(21,1);
% % % % for k = 1:21
% % % %     a(k) = f(k);
% % % % end
% % % % plot(a,':k')
% % % % grid
% % % % legend('Predicted','Database','Predicted trendline','Database trendline')
% % % % xlabel('Time (min)')
% % % % ylabel('API 30 min')
% % % % xtickangle(45);



%%%%%% nfle

% load('API_30s_16_nfle.mat')
% API_30s_16_nfle = [[0 0]; API_30s_16_nfle];
% figure
% plot(API_30s_16_nfle(:,1),'-r')
% hold all
% plot(API_30s_16_nfle(:,2),'--k')
% f=fit((2:1:length(API_30s_16_nfle(:,1)))',API_30s_16_nfle(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_30s_16_nfle(:,2)))',API_30s_16_nfle(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% ylabel('API 30 s')
% xlim([0 960])
% xtickangle(45);
% grid

% load('API_60s_16_nfle.mat')
% API_60s_16_nfle = [[0 0]; API_60s_16_nfle];
% figure
% plot(API_60s_16_nfle(:,1),'-r')
% hold all
% plot(API_60s_16_nfle(:,2),'--k')
% f=fit((2:1:length(API_60s_16_nfle(:,1)))',API_60s_16_nfle(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_60s_16_nfle(:,2)))',API_60s_16_nfle(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% ylabel('API 60 s')
% xlim([0 480])
% xtickangle(45);
% grid

% load('API_30m_16_nfle.mat')
% figure
% API_30m_16_nfle = [[0 0]; API_30m_16_nfle];
% plot(API_30m_16_nfle(:,1),'-r')
% hold all
% plot(API_30m_16_nfle(:,2),'--k')
% f=fit((2:1:length(API_30m_16_nfle(:,1)))',API_30m_16_nfle(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_30m_16_nfle(:,2)))',API_30m_16_nfle(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% ylabel('API 30 min')
% xtickangle(45);
% grid

% load('API_60m_16_nfle.mat')
% figure
% API_60m_16_nfle = [[0 0]; API_60m_16_nfle];
% plot(API_60m_16_nfle(:,1),'-r')
% hold all
% plot(API_60m_16_nfle(:,2),'--k')
% f=fit((2:1:length(API_60m_16_nfle(:,1)))',API_60m_16_nfle(2:end,1),'poly1');
% plot(f,'-.r')
% f=fit((2:1:length(API_60m_16_nfle(:,2)))',API_60m_16_nfle(2:end,2),'poly1');
% plot(f,':k')
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% ylabel('API 60 min')
% xtickangle(45);
% grid


% load('API_30s_16_nfle.mat')
% API_30s_16_nfle = [[0 0]; API_30s_16_nfle];
% figure
% plot(smoothdata(movmean(API_30s_16_nfle(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_30s_16_nfle(:,2),5)),'--k')
% f=fit((1:1:length(API_30s_16_nfle(:,1)))',smoothdata(movmean(API_30s_16_nfle(:,1),5)),'poly1');
% plot(f,'-.r')
% f=fit((1:1:length(API_30s_16_nfle(:,2)))',smoothdata(movmean(API_30s_16_nfle(:,2),5)),'poly1');
% plot(f,':k')
% grid
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% xlim([0 960])
% ylabel('API 30 s')
% xtickangle(45);

% load('API_60s_16_nfle.mat')
% API_60s_16_nfle = [[0 0]; API_60s_16_nfle];
% figure
% plot(smoothdata(movmean(API_60s_16_nfle(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_60s_16_nfle(:,2),5)),'--k')
% f=fit((1:1:length(API_60s_16_nfle(:,1)))',smoothdata(movmean(API_60s_16_nfle(:,1),5)),'poly1');
% plot(f,'-.r')
% f=fit((1:1:length(API_60s_16_nfle(:,2)))',smoothdata(movmean(API_60s_16_nfle(:,2),5)),'poly1');
% plot(f,':k')
% grid
% xlim([0 480])
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% ylabel('API 60 s')
% xtickangle(45);


% load('API_30m_16_nfle.mat')
% figure
% plot(smoothdata(movmean(API_30m_16_nfle(:,1),5)),'-r')
% hold all
% plot(smoothdata(movmean(API_30m_16_nfle(:,2),5)),'--k')
% f=fit((1:1:length(API_30m_16_nfle(:,1)))',smoothdata(movmean(API_30m_16_nfle(:,1),5)),'poly1');
% a=zeros(21,1);
% for k = 1:21
%     a(k) = f(k);
% end
% plot(a,'-.r')
% f=fit((1:1:length(API_30m_16_nfle(:,2)))',smoothdata(movmean(API_30m_16_nfle(:,2),5)),'poly1');
% a=zeros(21,1);
% for k = 1:21
%     a(k) = f(k);
% end
% plot(a,':k')
% grid
% legend('Predicted','Database','Predicted trendline','Database trendline')
% xlabel('Time (min)')
% ylabel('API 30 min')
% xtickangle(45);


load('API_60m_16_nfle.mat')
figure
plot(smoothdata(movmean(API_60m_16_nfle(:,1),5)),'-r')
hold all
plot(smoothdata(movmean(API_60m_16_nfle(:,2),5)),'--k')
f=fit((1:1:length(API_60m_16_nfle(:,1)))',smoothdata(movmean(API_60m_16_nfle(:,1),5)),'poly1');
plot(f,'-.r')
f=fit((1:1:length(API_60m_16_nfle(:,2)))',smoothdata(movmean(API_60m_16_nfle(:,2),5)),'poly1');
plot(f,':k')
grid
legend('Predicted','Database','Predicted trendline','Database trendline')
xlabel('Time (min)')
ylabel('API 60 min')
xtickangle(45);