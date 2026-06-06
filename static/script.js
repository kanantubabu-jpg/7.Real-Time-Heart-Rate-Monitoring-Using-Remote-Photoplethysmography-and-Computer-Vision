const bpmValue = document.getElementById('bpm-value');
const ctx = document.getElementById('bpmChart').getContext('2d');
let lastKnownBpm = null;

const chartData = {
    labels: [],
    datasets: [{
        label: 'BPM',
        data: [],
        borderColor: '#4ee1ff',
        backgroundColor: 'rgba(78, 225, 255, 0.25)',
        tension: 0.25,
        fill: true,
        pointRadius: 3,
        pointHoverRadius: 6,
    }],
};

const bpmChart = new Chart(ctx, {
    type: 'line',
    data: chartData,
    options: {
        responsive: true,
        scales: {
            x: {
                ticks: { color: '#cbd5e1' },
                grid: { color: 'rgba(255,255,255,0.08)' },
            },
            y: {
                min: 40,
                max: 130,
                ticks: { color: '#cbd5e1' },
                grid: { color: 'rgba(255,255,255,0.08)' },
            },
        },
        plugins: {
            legend: { display: false },
        },
    },
});

async function fetchBpm() {
    try {
        const res = await fetch('/status');
        const data = await res.json();
        const bpm = data.bpm;
        const quality = data.quality || 'Unknown';
        const samples = data.samples || 0;

        if (bpm !== null && bpm !== undefined) {
            lastKnownBpm = bpm;
            bpmValue.textContent = bpm;
            const timeLabel = new Date().toLocaleTimeString();
            chartData.labels.push(timeLabel);
            chartData.datasets[0].data.push(bpm);

            if (chartData.labels.length > 20) {
                chartData.labels.shift();
                chartData.datasets[0].data.shift();
            }
            bpmChart.update();
        } else if (lastKnownBpm !== null) {
            bpmValue.textContent = lastKnownBpm;
        } else if (quality === 'Collecting') {
            bpmValue.textContent = `Collecting (${samples})...`;
        } else {
            bpmValue.textContent = 'Measuring...';
        }
    } catch (error) {
        bpmValue.textContent = 'Error';
        console.error('Failed to load BPM data:', error);
    }
}

setInterval(fetchBpm, 400);
fetchBpm();
