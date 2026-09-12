document.addEventListener("DOMContentLoaded", () => {
    fetch('/api/model-info')
        .then(res => res.json())
        .then(data => {
            const status = document.getElementById('model-status');
            if (data.loaded) {
                status.textContent = `Model loaded: ${data.model_path} (${data.feature_count} features)`;
                status.style.color = '#4af626';
            } else {
                status.textContent = `Model failed to load`;
                status.style.color = '#d32f2f';
            }
        }).catch(err => {
            document.getElementById('model-status').textContent = 'Error connecting to API';
        });
});

async function analyzeCode() {
    const code = document.getElementById('code-input').value;
    const btn = document.getElementById('analyze-btn');
    const loading = document.getElementById('loading');
    const resultsContainer = document.getElementById('results-container');
    
    btn.disabled = true;
    loading.classList.remove('hidden');
    resultsContainer.innerHTML = '';
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            renderResults(data, resultsContainer);
        } else {
            resultsContainer.innerHTML = `<div style="color:red">Error: ${data.error || 'Failed to analyze'}</div>`;
        }
    } catch (err) {
        resultsContainer.innerHTML = `<div style="color:red">Network Error: ${err.message}</div>`;
    } finally {
        btn.disabled = false;
        loading.classList.add('hidden');
    }
}

function renderResults(data, container) {
    let loops = Array.isArray(data) ? data : (data.loops || []);
    let warnings = !Array.isArray(data) ? (data.warnings || []) : [];

    if (warnings.length > 0) {
        const warnDiv = document.createElement('div');
        warnDiv.style.backgroundColor = '#fff3cd';
        warnDiv.style.color = '#856404';
        warnDiv.style.padding = '10px';
        warnDiv.style.marginBottom = '15px';
        warnDiv.style.borderRadius = '4px';
        warnDiv.innerHTML = '<strong>Warnings:</strong><ul>' + warnings.map(w => `<li>${w}</li>`).join('') + '</ul>';
        container.appendChild(warnDiv);
    }

    if (!loops || loops.length === 0) {
        container.insertAdjacentHTML('beforeend', '<div class="placeholder">No candidate loops found.</div>');
        return;
    }
    
    const template = document.getElementById('loop-result-template');
    
    loops.forEach((loop, idx) => {
        const clone = template.content.cloneNode(true);
        
        // Header
        clone.querySelector('.loc-line').textContent = loop.loop_line;
        clone.querySelector('.loc-type').textContent = loop.loop_type;
        
        const badge = clone.querySelector('.verdict-badge');
        badge.textContent = `${loop.verdict} (${(loop.confidence*100).toFixed(0)}%)`;
        badge.classList.add(loop.verdict);
        
        // Snippet
        clone.querySelector('.source-snippet').textContent = loop.source_snippet;
        
        // Diagnostic
        clone.querySelector('.diagnostic-output').textContent = loop.diagnostic;
        
        // Explanation
        clone.querySelector('.explanation-box').textContent = loop.explanation;
        
        // Fixes
        if (loop.fix_suggestions && loop.fix_suggestions.length > 0) {
            const fixContainer = clone.querySelector('.fix-suggestions-container');
            fixContainer.classList.remove('hidden');
            const fixList = clone.querySelector('.fix-list');
            loop.fix_suggestions.forEach(fix => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <div><strong>Problem:</strong> ${fix.problem}</div>
                    <div><strong>Fix:</strong> ${fix.fix}</div>
                    <div style="font-size:0.9em; margin-top:0.3rem;"><em>Expect: ${fix.expected_improvement}</em></div>
                    ${fix.code_example ? `<span class="fix-example">${fix.code_example}</span>` : ''}
                `;
                fixList.appendChild(li);
            });
        }
        
        // Roofline
        if (loop.roofline_position) {
            clone.querySelector('.rfl-bound').textContent = loop.roofline_position.boundedness || 'unknown';
            clone.querySelector('.rfl-ai').textContent = (loop.roofline_position.arithmetic_intensity || 0).toFixed(2);
            clone.querySelector('.rfl-gflops').textContent = (loop.roofline_position.estimated_gflops || 0).toFixed(1);
        }
        
        // Charts
        const shapCanvas = clone.querySelector('.shap-chart');
        const crossCanvas = clone.querySelector('.crossover-chart');
        
        // Append first so we can draw charts on it
        container.appendChild(clone);
        
        drawShapChart(shapCanvas, loop.grouped_shap);
        
        if (loop.profitability_curve && loop.profitability_curve.length > 0) {
            drawCrossoverChart(crossCanvas, loop.profitability_curve, loop.crossover_n);
        } else {
            crossCanvas.parentElement.innerHTML = '<h4>Profitability Crossover</h4><p style="text-align:center;color:#888;">Not applicable for blocked/unpredictable loops</p>';
        }
    });
}

function drawShapChart(canvas, groupedShap) {
    if (!groupedShap) return;
    
    const labels = Object.keys(groupedShap);
    const data = Object.values(groupedShap);
    const colors = data.map(v => v > 0 ? 'rgba(74, 144, 226, 0.8)' : 'rgba(211, 47, 47, 0.8)');
    
    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'SHAP Value (Impact on Profitability)',
                data: data,
                backgroundColor: colors
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { grid: { color: '#444' }, ticks: { color: '#ccc' } },
                y: { grid: { display: false }, ticks: { color: '#ccc', font: {size: 10} } }
            }
        }
    });
}

function drawCrossoverChart(canvas, curve, crossoverN) {
    if (!curve || curve.length === 0) return;
    
    const labels = curve.map(pt => pt.N);
    const probs = curve.map(pt => pt.probability);
    
    new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'P(Profitable)',
                data: probs,
                borderColor: '#4af626',
                backgroundColor: 'rgba(74, 246, 38, 0.1)',
                fill: true,
                tension: 0.2
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: { 
                    type: 'logarithmic',
                    title: { display: true, text: 'Problem Size (N)', color: '#ccc' },
                    grid: { color: '#333' },
                    ticks: { color: '#888' }
                },
                y: { 
                    min: 0, max: 1.0,
                    title: { display: true, text: 'Probability', color: '#ccc' },
                    grid: { color: '#444' },
                    ticks: { color: '#888' }
                }
            },
            plugins: {
                annotation: crossoverN ? {
                    annotations: {
                        line1: {
                            type: 'line',
                            xMin: crossoverN,
                            xMax: crossoverN,
                            borderColor: 'red',
                            borderWidth: 2,
                            label: { content: 'Breakeven', enabled: true }
                        }
                    }
                } : {}
            }
        }
    });
}
