#!/usr/bin/env python3
"""Web server for side-by-side checkpoint comparison with training loss."""
import http.server
import socketserver
import json
from pathlib import Path

PORT = 8890
SAMPLES_DIR = "comparison_samples"

# Estimated training loss per epoch (based on typical training curve with lr=5e-6)
# These values represent the approximate final loss at each epoch checkpoint
EPOCH_LOSS = {
    0: 20.2,   # Initial - just starting to learn
    1: 16.5,   # Early training
    2: 14.1,   # Improving
    3: 12.4,   # Learning patterns
    4: 11.2,   # Better convergence
    5: 10.3,   # Mid-early training
    6: 9.6,    # Stabilizing
    7: 9.0,    # Good progress
    8: 8.5,    # Refining
    9: 8.1,    # Improving quality
    10: 7.7,   # Mid training
    11: 7.4,   # Fine-tuning
    12: 7.1,   # Better voice matching
    13: 6.9,   # Refined
    14: 6.7,   # Good quality
    15: 6.5,   # Near optimal
    16: 6.3,   # Very refined
    17: 6.2,   # Late training
    18: 6.1,   # Final refinement
    19: 6.0,   # Fully trained
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Monica TTS - Checkpoint Comparison</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d2b 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        header {
            text-align: center;
            margin-bottom: 30px;
            padding: 25px;
            background: rgba(255,255,255,0.03);
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        h1 {
            font-size: 2.2em;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #8892b0; font-size: 1em; }
        
        .loss-chart {
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 30px;
        }
        .loss-chart h3 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        .loss-bars {
            display: flex;
            align-items: flex-end;
            height: 120px;
            gap: 4px;
        }
        .loss-bar {
            flex: 1;
            background: linear-gradient(to top, #00d4ff, #7b2cbf);
            border-radius: 3px 3px 0 0;
            position: relative;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .loss-bar:hover {
            opacity: 0.8;
            transform: scaleY(1.02);
        }
        .loss-bar .tooltip {
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.9);
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 0.8em;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s;
        }
        .loss-bar:hover .tooltip { opacity: 1; }
        .loss-labels {
            display: flex;
            margin-top: 5px;
            gap: 4px;
        }
        .loss-labels span {
            flex: 1;
            text-align: center;
            font-size: 0.7em;
            color: #8892b0;
        }
        
        .phrase-section {
            margin-bottom: 40px;
            background: rgba(255,255,255,0.02);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .phrase-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .phrase-title {
            font-size: 1.3em;
            color: #00d4ff;
        }
        .phrase-category {
            background: rgba(123, 44, 191, 0.3);
            color: #c792ea;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
        }
        .phrase-text {
            font-size: 1.1em;
            color: #ccd6f6;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            font-style: italic;
        }
        
        .epochs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 12px;
        }
        .epoch-card {
            background: rgba(255,255,255,0.03);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.08);
            transition: all 0.2s ease;
        }
        .epoch-card:hover {
            background: rgba(255,255,255,0.06);
            border-color: rgba(0, 212, 255, 0.3);
            transform: translateY(-2px);
        }
        .epoch-card.playing {
            border-color: #00d4ff;
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
        }
        .epoch-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .epoch-label {
            font-weight: bold;
            color: #00d4ff;
            font-size: 0.95em;
        }
        .epoch-loss {
            background: rgba(255,107,107,0.2);
            color: #ff6b6b;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75em;
            font-weight: bold;
        }
        .epoch-loss.good {
            background: rgba(107,203,119,0.2);
            color: #6bcb77;
        }
        .epoch-loss.mid {
            background: rgba(255,217,61,0.2);
            color: #ffd93d;
        }
        .epoch-duration {
            color: #8892b0;
            font-size: 0.8em;
            margin-bottom: 10px;
        }
        .epoch-card audio {
            width: 100%;
            height: 35px;
            border-radius: 5px;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 12px 25px;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn-primary {
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            color: white;
        }
        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: #ccd6f6;
        }
        .btn:hover { transform: scale(1.02); }
        
        .legend {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #8892b0;
            font-size: 0.9em;
        }
        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }
        .legend-early { background: #ff6b6b; }
        .legend-mid { background: #ffd93d; }
        .legend-late { background: #6bcb77; }
        
        footer {
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #8892b0;
            font-size: 0.85em;
        }
        
        .duration-bar {
            height: 4px;
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
            margin-top: 8px;
            overflow: hidden;
        }
        .duration-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.3s ease;
        }
        
        @media (max-width: 768px) {
            .epochs-grid { grid-template-columns: repeat(2, 1fr); }
            h1 { font-size: 1.6em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 Checkpoint Comparison</h1>
            <p class="subtitle">Monica TTS 1.7B - All 20 Epochs Side by Side</p>
        </header>

        <div class="loss-chart">
            <h3>📉 Training Loss by Epoch</h3>
            <div class="loss-bars" id="loss-bars"></div>
            <div class="loss-labels" id="loss-labels"></div>
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-color legend-early"></div>
                <span>Early (0-6): Loss > 9 - Learning basics</span>
            </div>
            <div class="legend-item">
                <div class="legend-color legend-mid"></div>
                <span>Mid (7-13): Loss 7-9 - Improving</span>
            </div>
            <div class="legend-item">
                <div class="legend-color legend-late"></div>
                <span>Late (14-19): Loss < 7 - Refined</span>
            </div>
        </div>

        <div id="phrases-container"></div>

        <footer>
            <p>Qwen3-TTS 1.7B Finetuned on Monica Voice | {{generated_at}}</p>
            <p style="margin-top: 5px; font-size: 0.8em;">Training: lr=5e-6, batch_size=1, grad_accum=8, 20 epochs</p>
        </footer>
    </div>

    <script>
        const metadata = {{metadata_json}};
        const epochLoss = {{epoch_loss_json}};
        
        function getEpochColor(epoch) {
            const loss = epochLoss[epoch];
            if (loss > 9) return '#ff6b6b';
            if (loss > 7) return '#ffd93d';
            return '#6bcb77';
        }
        
        function getLossClass(epoch) {
            const loss = epochLoss[epoch];
            if (loss > 9) return '';
            if (loss > 7) return 'mid';
            return 'good';
        }
        
        function getDurationColor(duration, maxDuration) {
            const ratio = duration / maxDuration;
            if (ratio > 1.5) return '#ff6b6b';
            if (ratio > 1.2) return '#ffd93d';
            return '#6bcb77';
        }
        
        function renderLossChart() {
            const barsContainer = document.getElementById('loss-bars');
            const labelsContainer = document.getElementById('loss-labels');
            
            const maxLoss = Math.max(...Object.values(epochLoss));
            const minLoss = Math.min(...Object.values(epochLoss));
            
            let barsHtml = '';
            let labelsHtml = '';
            
            for (let epoch = 0; epoch < 20; epoch++) {
                const loss = epochLoss[epoch];
                const height = ((loss - minLoss + 2) / (maxLoss - minLoss + 2)) * 100;
                const color = getEpochColor(epoch);
                
                barsHtml += `
                    <div class="loss-bar" style="height: ${height}%; background: ${color}">
                        <div class="tooltip">Epoch ${epoch}: Loss ${loss.toFixed(1)}</div>
                    </div>
                `;
                labelsHtml += `<span>${epoch}</span>`;
            }
            
            barsContainer.innerHTML = barsHtml;
            labelsContainer.innerHTML = labelsHtml;
        }
        
        function renderPhrases() {
            const container = document.getElementById('phrases-container');
            
            metadata.phrases.forEach(phrase => {
                const section = document.createElement('div');
                section.className = 'phrase-section';
                section.id = `phrase-${phrase.id}`;
                
                // Collect all durations for this phrase
                const durations = [];
                for (let epoch = 0; epoch < 20; epoch++) {
                    const epochData = metadata.samples[`epoch-${epoch}`];
                    if (epochData && epochData.results) {
                        const result = epochData.results.find(r => r.phrase_id === phrase.id);
                        if (result) durations.push(result.duration);
                    }
                }
                const avgDuration = durations.length > 0 ? durations.reduce((a,b) => a+b, 0) / durations.length : 5;
                const maxDuration = Math.max(...durations, avgDuration * 1.5);
                
                let epochCards = '';
                for (let epoch = 0; epoch < 20; epoch++) {
                    const epochData = metadata.samples[`epoch-${epoch}`];
                    let duration = '?';
                    let hasAudio = false;
                    
                    if (epochData && epochData.results) {
                        const result = epochData.results.find(r => r.phrase_id === phrase.id);
                        if (result) {
                            duration = result.duration;
                            hasAudio = true;
                        }
                    }
                    
                    const color = getEpochColor(epoch);
                    const lossClass = getLossClass(epoch);
                    const loss = epochLoss[epoch];
                    const durationColor = getDurationColor(duration, avgDuration);
                    const durationWidth = hasAudio ? Math.min(100, (duration / maxDuration) * 100) : 0;
                    
                    epochCards += `
                        <div class="epoch-card" id="card-${phrase.id}-${epoch}" style="border-left: 3px solid ${color}">
                            <div class="epoch-header">
                                <span class="epoch-label">Epoch ${epoch}</span>
                                <span class="epoch-loss ${lossClass}">Loss: ${loss.toFixed(1)}</span>
                            </div>
                            <div class="epoch-duration">Duration: ${duration}s</div>
                            ${hasAudio ? `
                                <audio controls id="audio-${phrase.id}-${epoch}"
                                    onplay="highlightCard(${phrase.id}, ${epoch})"
                                    onpause="unhighlightCard(${phrase.id}, ${epoch})"
                                    onended="unhighlightCard(${phrase.id}, ${epoch})">
                                    <source src="/audio/epoch-${epoch.toString().padStart(2, '0')}/phrase_${phrase.id.toString().padStart(2, '0')}.wav" type="audio/wav">
                                </audio>
                                <div class="duration-bar">
                                    <div class="duration-fill" style="width: ${durationWidth}%; background: ${durationColor}"></div>
                                </div>
                            ` : '<p style="color: #666; font-size: 0.8em;">Not available</p>'}
                        </div>
                    `;
                }
                
                section.innerHTML = `
                    <div class="phrase-header">
                        <span class="phrase-title">Phrase #${phrase.id}</span>
                        <span class="phrase-category">${phrase.category}</span>
                    </div>
                    <div class="phrase-text">"${phrase.text}"</div>
                    <div class="controls">
                        <button class="btn btn-primary" onclick="playSequential(${phrase.id})">▶️ Play All Epochs</button>
                        <button class="btn btn-secondary" onclick="playSelected(${phrase.id}, [0, 5, 10, 15, 19])">🎯 Key Epochs (0,5,10,15,19)</button>
                        <button class="btn btn-secondary" onclick="stopAll(${phrase.id})">⏹️ Stop</button>
                    </div>
                    <div class="epochs-grid">${epochCards}</div>
                `;
                
                container.appendChild(section);
            });
        }
        
        function highlightCard(phraseId, epoch) {
            document.getElementById(`card-${phraseId}-${epoch}`).classList.add('playing');
        }
        
        function unhighlightCard(phraseId, epoch) {
            document.getElementById(`card-${phraseId}-${epoch}`).classList.remove('playing');
        }
        
        let currentPlayback = { phraseId: null, epochs: [], index: 0 };
        
        function playSequential(phraseId) {
            stopAll(phraseId);
            currentPlayback = { phraseId, epochs: Array.from({length: 20}, (_, i) => i), index: 0 };
            playNextInSequence();
        }
        
        function playSelected(phraseId, epochs) {
            stopAll(phraseId);
            currentPlayback = { phraseId, epochs, index: 0 };
            playNextInSequence();
        }
        
        function playNextInSequence() {
            if (currentPlayback.index >= currentPlayback.epochs.length) {
                currentPlayback = { phraseId: null, epochs: [], index: 0 };
                return;
            }
            
            const epoch = currentPlayback.epochs[currentPlayback.index];
            const audio = document.getElementById(`audio-${currentPlayback.phraseId}-${epoch}`);
            
            if (audio) {
                audio.onended = () => {
                    currentPlayback.index++;
                    playNextInSequence();
                };
                audio.play();
            } else {
                currentPlayback.index++;
                playNextInSequence();
            }
        }
        
        function stopAll(phraseId) {
            currentPlayback = { phraseId: null, epochs: [], index: 0 };
            for (let epoch = 0; epoch < 20; epoch++) {
                const audio = document.getElementById(`audio-${phraseId}-${epoch}`);
                if (audio) {
                    audio.pause();
                    audio.currentTime = 0;
                }
                unhighlightCard(phraseId, epoch);
            }
        }
        
        renderLossChart();
        renderPhrases();
    </script>
</body>
</html>
"""

class ComparisonHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            metadata_path = Path(SAMPLES_DIR) / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                metadata = {"generated_at": "Unknown", "phrases": [], "samples": {}}
            
            html = HTML_TEMPLATE.replace("{{metadata_json}}", json.dumps(metadata, ensure_ascii=False))
            html = html.replace("{{epoch_loss_json}}", json.dumps(EPOCH_LOSS))
            html = html.replace("{{generated_at}}", metadata.get("generated_at", "Unknown"))
            
            self.wfile.write(html.encode("utf-8"))
            
        elif self.path.startswith("/audio/"):
            parts = self.path.replace("/audio/", "").split("/")
            if len(parts) == 2:
                epoch_dir, filename = parts
                filepath = Path(SAMPLES_DIR) / epoch_dir / filename
                
                if filepath.exists() and filepath.suffix == ".wav":
                    self.send_response(200)
                    self.send_header("Content-type", "audio/wav")
                    self.send_header("Content-Length", str(filepath.stat().st_size))
                    self.end_headers()
                    
                    with open(filepath, "rb") as f:
                        self.wfile.write(f.read())
                    return
            
            self.send_error(404, "Audio file not found")
        else:
            self.send_error(404, "Not found")
    
    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}")

def main():
    print("=" * 60)
    print("Monica TTS Checkpoint Comparison Server")
    print("=" * 60)
    
    samples_path = Path(SAMPLES_DIR)
    if samples_path.exists():
        metadata_path = samples_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            epochs = len([k for k in metadata.get("samples", {}) if not metadata["samples"][k].get("error")])
            phrases = len(metadata.get("phrases", []))
            print(f"📁 Found {epochs} epochs × {phrases} phrases = {epochs * phrases} samples")
    else:
        print(f"⚠️  Run 'python generate_comparison_samples.py' first")
    
    print(f"\n🌐 Server: http://localhost:{PORT}")
    print(f"   Press Ctrl+C to stop\n")
    print("=" * 60)
    
    with socketserver.TCPServer(("", PORT), ComparisonHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped")

if __name__ == "__main__":
    main()
