// static/chart-client.js
(function ()
{

    const chartDiv = document.getElementById('tvchart');
    const tooltip = document.getElementById('tooltip');
    const noDataElem = document.getElementById('no-data');

    // ============================================
    // 1. VISUAL CONFIGURATION (Exact NSE Style)
    // ============================================
    const CONFIG = {
        colors: {
            line: '#008CA3',            // NSE Teal
            topFill: 'rgba(0, 140, 163, 0.2)',
            bottomFill: 'rgba(0, 140, 163, 0.0)',
            prevCloseLine: '#888888',   // Grey reference line
            text: '#333333',
            grid: 'rgba(0, 0, 0, 0.02)', // Very subtle grid
            bg: '#ffffff',
            crosshair: '#555555'
        },
        // Fonts matching the OS native look of the screenshot
        font: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    };

    // ============================================
    // 2. CREATE CHART INSTANCE
    // ============================================
    const chart = LightweightCharts.createChart(chartDiv, {
        width: chartDiv.clientWidth,
        height: chartDiv.clientHeight,
        layout: {
            background: { color: CONFIG.colors.bg },
            textColor: CONFIG.colors.text,
            fontSize: 11,
            fontFamily: CONFIG.font,
        },
        grid: {
            vertLines: { color: CONFIG.colors.grid },
            horzLines: { color: CONFIG.colors.grid }
        },
        // LEFT PRICE SCALE (Matches AA.png)
        leftPriceScale: {
            visible: true,
            borderColor: 'transparent', // Hide the border line itself for a cleaner look
            scaleMargins: {
                top: 0.1,    // Leave space at top
                bottom: 0.1, // Leave space at bottom
            },
        },
        rightPriceScale: {
            visible: false, // Hide right scale
        },
        timeScale: {
            borderColor: '#E0E0E0',
            timeVisible: true,
            secondsVisible: false,
            fixLeftEdge: true,
            fixRightEdge: true,
            rightOffset: 10,
            tickMarkFormatter: (time) =>
            {
                const d = new Date(time * 1000);
                // Return HH:mm format
                return d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
            }
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: {
                color: CONFIG.colors.crosshair,
                width: 1,
                style: 1, // Dotted
                labelBackgroundColor: CONFIG.colors.line,
            },
            horzLine: {
                color: CONFIG.colors.crosshair,
                width: 1,
                style: 1, // Dotted
                labelBackgroundColor: CONFIG.colors.line,
            }
        },
        handleScale: false, // Disable scaling for a fixed "app-like" feel
        handleScroll: false // Disable scrolling to mimic the static view
    });

    // ============================================
    // 3. ADD AREA SERIES
    // ============================================
    const areaSeries = chart.addAreaSeries({
        topColor: CONFIG.colors.topFill,
        bottomColor: CONFIG.colors.bottomFill,
        lineColor: CONFIG.colors.line,
        lineWidth: 2,
        priceScaleId: 'left', // IMPORTANT: Bind to left scale
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 4,
        crosshairMarkerBorderColor: '#ffffff',
        crosshairMarkerBackgroundColor: CONFIG.colors.line,
    });

    // ============================================
    // 4. MOCK DATA GENERATOR (To match AA.png)
    // ============================================
    // This function creates the specific curve: Dip -> Rise -> Flatten
    function generateNSELikeData ()
    {
        const data = [];

        // Start Time: 9:15 AM
        let currentTime = new Date();
        currentTime.setHours(9, 15, 0, 0);
        let timeUnix = Math.floor(currentTime.getTime() / 1000);

        // Parameters matching the screenshot
        let price = 527.45; // Open
        const prevClose = 528.75;
        const targetHigh = 555.15;
        const dayLow = 522.65;

        // Simulating 375 minutes (9:15 to 15:30)
        for (let i = 0; i <= 375; i++)
        {

            // 1. Morning Dip (First 30 mins)
            if (i < 30)
            {
                price -= (Math.random() * 0.5);
                if (price < dayLow) price = dayLow;
            }
            // 2. Steady Rise (Next 3 hours)
            else if (i < 240)
            {
                price += (Math.random() * 0.4); // Bias upwards
                // Add some noise
                price += (Math.random() - 0.5) * 0.5;
            }
            // 3. Afternoon Plateau (Last 2 hours)
            else
            {
                // Stay near high but fluctuate slightly
                let drift = (Math.random() - 0.5) * 0.8;
                price += drift;
                // Soft clamp to near 550
                if (price > 552) price -= 0.2;
                if (price < 548) price += 0.2;
            }

            data.push({ time: timeUnix + (i * 60), value: price });
        }

        return { data, prevClose };
    }

    // ============================================
    // 5. RENDER LOGIC
    // ============================================

    // Draw the "Previous Close" Reference Line
    function drawPrevCloseLine (price)
    {
        // Create a separate line series for the reference line to control it independently
        // Or use the built-in PriceLine feature on the area series
        const line = areaSeries.createPriceLine({
            price: price,
            color: CONFIG.colors.prevCloseLine,
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Solid, // NSE uses a solid grey line usually
            axisLabelVisible: true,
            title: 'Prev Close',
        });
    }

    function init ()
    {
        // 1. Get Data (Using the generator to ensure it looks like your image)
        const { data, prevClose } = generateNSELikeData();

        if (data.length > 0)
        {
            // 2. Set Data
            areaSeries.setData(data);

            // 3. Add Previous Close Line
            drawPrevCloseLine(prevClose);

            // 4. Fit Content
            chart.timeScale().fitContent();

            // 5. Hide "No Data" Label
            if (noDataElem) noDataElem.style.display = 'none';
        }
    }

    // ============================================
    // 6. CUSTOM TOOLTIP (Floating)
    // ============================================
    chart.subscribeCrosshairMove((param) =>
    {
        if (!param.time || !param.point || param.point.x < 0 || param.point.x > chartDiv.clientWidth || param.point.y < 0 || param.point.y > chartDiv.clientHeight)
        {
            tooltip.style.display = "none";
            return;
        }

        const price = param.seriesPrices.get(areaSeries);
        const d = new Date(param.time * 1000);
        const timeStr = d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
        const dateStr = d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' });

        tooltip.style.left = (param.point.x + 10) + "px";
        tooltip.style.top = (param.point.y + 10) + "px";
        tooltip.style.display = "block";

        // Exact NSE Tooltip styling
        tooltip.innerHTML = `
            <div style="font-size: 11px; color: #666; margin-bottom: 2px;">${dateStr} | ${timeStr}</div>
            <div style="font-size: 14px; font-weight: 700; color: #333;">
                ₹${price.toFixed(2)}
            </div>
        `;
    });

    // ============================================
    // 7. RESPONSIVENESS
    // ============================================
    window.addEventListener("resize", () =>
    {
        chart.applyOptions({
            width: chartDiv.clientWidth,
            height: chartDiv.clientHeight
        });
    });

    // Run
    init();

})();