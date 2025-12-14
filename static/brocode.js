document.addEventListener("DOMContentLoaded", async () =>
{

  const chartContainer = document.getElementById("tvchart");

  const chart = LightweightCharts.createChart(chartContainer, {
    width: chartContainer.clientWidth,
    height: 400,
    layout: {
      background: { type: 'solid', color: '#ffffff' },
      textColor: '#000',
    },
    grid: {
      vertLines: { color: '#eeeeee' },
      horzLines: { color: '#eeeeee' },
    },
    timeScale: {
      timeVisible: true,
      secondsVisible: false,
    },
    rightPriceScale: {
      borderColor: "#cccccc",
    },
  });

  const areaSeries = chart.addAreaSeries({
    topColor: 'rgba(0, 150, 255, 0.5)',
    bottomColor: 'rgba(0, 200, 255, 0.1)',
    lineColor: '#0096ff',
    lineWidth: 2,
  });

  // Fetch Data from API
  async function loadChartData ()
  {
    try
    {
      const res = await fetch(`/chart-data/${symbol}`);
      const data = await res.json();

      const formatted = data.map(d => ({
        time: Math.floor(new Date(d.t).getTime() / 1000),
        value: d.c
      }));

      areaSeries.setData(formatted);
      chart.timeScale().fitContent();

    } catch (err)
    {
      console.error("Chart Fetch Error:", err);
    }
  }

  await loadChartData();

  // Resize chart on window change
  const resize = () =>
  {
    chart.applyOptions({ width: chartContainer.clientWidth });
  };
  window.addEventListener('resize', resize);

});