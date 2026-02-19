/**
 * OtoWake ONNX — ランディングページ統合アプリケーション
 *
 * すべての処理がブラウザ内で完結する。
 * バックエンド API / WebSocket への依存は一切なし。
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { MODEL_CONFIGS, DEFAULT_MODEL } from './lib/model-config.ts';
import {
  decodeAudioFile,
  resampleAudio,
  audioBufferToChannelData,
  encodeWav,
} from './lib/audio-utils.ts';
import { OnnxSeparator } from './lib/onnx-engine.ts';
import type { StemResult } from './lib/onnx-engine.ts';

type ConsoleState = 'idle' | 'uploaded' | 'processing' | 'completed' | 'error';

interface DownloadableTrack {
  name: string;
  fileName: string;
  url: string;
  blob: Blob;
}

const ALL_PARTS = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'] as const;

const PART_LABELS: Record<string, string> = {
  vocals: 'Vocals',
  drums: 'Drums',
  bass: 'Bass',
  guitar: 'Guitar',
  piano: 'Piano',
  other: 'Other',
};

const FAQ_ITEMS = [
  { q: '対応ファイル形式は？', a: 'MP3、WAV、FLAC、OGGに対応しています。' },
  { q: '処理時間は？', a: '3〜5分の楽曲で数十秒〜数分。GPU搭載PCではさらに高速です。' },
  { q: '音源がサーバーに送られる？', a: 'いいえ。すべてPC上で完結します。' },
  { q: 'スマホでも使える？', a: '現在はPC（macOS / Windows）向けです。' },
];

export default function App() {
  /* ---- 状態 ---- */
  const [consoleState, setConsoleState] = useState<ConsoleState>('idle');
  const [file, setFile] = useState<File | null>(null);
  const [removeMode, setRemoveMode] = useState(false);
  const [selectedParts, setSelectedParts] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const [tracks, setTracks] = useState<DownloadableTrack[]>([]);
  const [dragover, setDragover] = useState(false);
  const [openFaq, setOpenFaq] = useState<number | null>(null);

  const [elapsedSec, setElapsedSec] = useState(0);

  const separatorRef = useRef(new OnnxSeparator());
  const fileInputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* ---- スクロールフェードイン ---- */
  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) e.target.classList.add('visible');
        });
      },
      { threshold: 0.1 },
    );
    document.querySelectorAll('.fade-in').forEach((el) => obs.observe(el));
    return () => obs.disconnect();
  }, []);

  /* ---- ファイル選択 ---- */
  const handleFile = useCallback((f: File) => {
    for (const t of tracks) URL.revokeObjectURL(t.url);
    setFile(f);
    setTracks([]);
    setConsoleState('uploaded');
    setErrorMsg('');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragover(false);
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    },
    [handleFile],
  );

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files?.length) handleFile(e.target.files[0]);
    },
    [handleFile],
  );

  /* ---- パート切替 ---- */
  const togglePart = (part: string) => {
    setSelectedParts((prev) =>
      prev.includes(part) ? prev.filter((p) => p !== part) : [...prev, part],
    );
  };

  /* ---- 分離処理 ---- */
  const handleRun = async () => {
    if (!file) return;
    if (selectedParts.length === 0) {
      setErrorMsg('少なくとも1つのパートを選択してください');
      return;
    }

    const config = MODEL_CONFIGS[DEFAULT_MODEL];
    setConsoleState('processing');
    setProgress(0);
    setProgressMsg('初期化中...');
    setErrorMsg('');
    setElapsedSec(0);
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => setElapsedSec((s) => s + 1), 1000);

    try {
      const separator = separatorRef.current;

      // 1. モデル読み込み
      if (separator.getLoadedModelName() !== config.name) {
        await separator.loadModel(config, (p, msg) => {
          setProgress(Math.round(p * 0.3));
          setProgressMsg(msg);
        });
      }

      // 2. 音声デコード
      setProgress(30);
      setProgressMsg('音声ファイルをデコード中...');
      const audioBuffer = await decodeAudioFile(file);

      // 3. リサンプル
      setProgress(35);
      setProgressMsg('リサンプル中...');
      const resampled = await resampleAudio(audioBuffer, config.sampleRate);

      // 4. チャネルデータ変換
      const { data, channels, length } = audioBufferToChannelData(resampled);

      // 5. ONNX 分離実行
      const result = await separator.separate(
        data,
        channels,
        length,
        selectedParts,
        removeMode,
        (p, msg) => {
          setProgress(40 + Math.round(p * 0.5));
          setProgressMsg(msg);
        },
      );

      // 6. WAV エンコード
      setProgress(92);
      setProgressMsg('WAVファイルを生成中...');
      await new Promise((r) => setTimeout(r, 0));

      const baseName = file.name.replace(/\.[^.]+$/, '');
      const downloadable = result.stems.map((stem: StemResult) => {
        const blob = encodeWav(stem.channels, result.sampleRate);
        const fileName = `${baseName}_${stem.name}.wav`;
        return { name: stem.name, fileName, url: URL.createObjectURL(blob), blob };
      });

      setTracks(downloadable);
      setProgress(100);
      setProgressMsg('分離完了!');
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      setConsoleState('completed');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      setConsoleState('error');
      setErrorMsg(message);
    }
  };

  /* ---- リセット ---- */
  const handleReset = () => {
    for (const t of tracks) URL.revokeObjectURL(t.url);
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    setConsoleState('idle');
    setFile(null);
    setSelectedParts([]);
    setRemoveMode(false);
    setTracks([]);
    setErrorMsg('');
    setProgress(0);
    setProgressMsg('');
    setElapsedSec(0);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  /* ---- FAQ ---- */
  const toggleFaq = (idx: number) => {
    setOpenFaq((prev) => (prev === idx ? null : idx));
  };

  const isProcessing = consoleState === 'processing';
  const hasFile = consoleState !== 'idle';

  const formatElapsed = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  /* ================================================================ */

  return (
    <>
      {/* ===== NAV ===== */}
      <nav className="nav">
        <div className="nav-logo">OtoWake</div>
        <ul className="nav-links">
          <li><a href="#howto">使い方</a></li>
          <li><a href="#recommend">こんな人に</a></li>
          <li><a href="#features">特徴</a></li>
          <li><a href="#faq">よくある質問</a></li>
        </ul>
      </nav>

      {/* ===== HERO ===== */}
      <section className="hero" id="hero">
        <p className="hero-logo">OtoWake</p>
        <h1 className="hero-title">好きな曲で、歌ってみよう。弾いてみよう。</h1>
        <p className="hero-sub">ボーカル除去もパート抽出も、ブラウザだけで完結。</p>

        <div className="console">
          {/* ドロップエリア / ファイルバー */}
          {!hasFile ? (
            <div
              className={`drop-area${dragover ? ' dragover' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
              onDragLeave={() => setDragover(false)}
              onDrop={onDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={onFileChange}
              />
              <svg viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <p className="drop-label">音源をドロップ、または<strong>選択</strong></p>
              <p className="drop-hint">MP3 / WAV / FLAC / OGG</p>
            </div>
          ) : (
            <div className="file-bar">
              <svg viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round">
                <path d="M9 18V5l12-2v13" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="16" r="3" />
              </svg>
              <span className="file-bar-name">{file?.name}</span>
              {!isProcessing && (
                <button className="file-bar-remove" onClick={handleReset}>&times;</button>
              )}
            </div>
          )}

          {/* コントロール */}
          <div className={`console-controls${!hasFile || isProcessing ? ' disabled' : ''}`}>
            <div className="mode-row">
              <button
                className={`mode-btn${!removeMode ? ' active' : ''}`}
                onClick={() => setRemoveMode(false)}
              >
                抽出する
              </button>
              <button
                className={`mode-btn${removeMode ? ' active' : ''}`}
                onClick={() => setRemoveMode(true)}
              >
                除去する
              </button>
            </div>
            <div className="parts-row">
              {ALL_PARTS.map((part) => (
                <button
                  key={part}
                  className={`part-btn${selectedParts.includes(part) ? ' active' : ''}`}
                  onClick={() => togglePart(part)}
                >
                  {PART_LABELS[part]}
                </button>
              ))}
            </div>
            <button
              className="btn-run"
              disabled={isProcessing || selectedParts.length === 0}
              onClick={handleRun}
            >
              {isProcessing ? '処理中...' : '分離する'}
            </button>
          </div>

          {/* プログレス */}
          {isProcessing && (
            <div className="progress-area">
              <div className="progress-outer">
                <div className="progress-inner" style={{ width: `${progress}%` }}>
                  <div className="progress-shimmer" />
                </div>
              </div>
              <div className="progress-detail">
                <span className="progress-msg">
                  <span className="spinner" />
                  {progressMsg || '処理開始中...'}
                </span>
                <span className="progress-elapsed">{formatElapsed(elapsedSec)}</span>
              </div>
            </div>
          )}

          {/* 結果 */}
          {consoleState === 'completed' && tracks.length > 0 && (
            <div className="results-area">
              <ul className="results-list">
                {tracks.map((t) => (
                  <li key={t.name}>
                    <span className="track-name">{t.name}</span>
                    <a className="dl-btn" href={t.url} download={t.fileName}>ダウンロード</a>
                  </li>
                ))}
              </ul>
              <button className="btn-reset" onClick={handleReset}>
                新しいファイルを処理
              </button>
            </div>
          )}

          {/* エラー */}
          {(consoleState === 'error' && errorMsg) && (
            <div className="error-msg">{errorMsg}</div>
          )}

          <p className="console-note">無料 / 登録不要</p>
        </div>
      </section>

      {/* ===== HOWTO ===== */}
      <section className="howto fade-in" id="howto">
        <div className="wrap">
          <div className="howto-header">
            <p className="section-label">How it works</p>
            <h2 className="section-heading">使い方</h2>
          </div>
          <div className="howto-steps">
            <div className="howto-card glass">
              <span className="step-tag">STEP 01</span>
              <h3>音源を選択</h3>
              <p>ファイルをドロップ</p>
            </div>
            <div className="howto-card glass">
              <span className="step-tag">STEP 02</span>
              <h3>モードとパートを選ぶ</h3>
              <p>抽出 or 除去</p>
            </div>
            <div className="howto-card glass">
              <span className="step-tag">STEP 03</span>
              <h3>ダウンロード</h3>
              <p>分離された音源を保存</p>
            </div>
          </div>
        </div>
      </section>

      {/* ===== RECOMMEND ===== */}
      <section className="recommend fade-in" id="recommend">
        <div className="wrap">
          <div className="recommend-header">
            <p className="section-label">Use Cases</p>
            <h2 className="section-heading">こんな人に</h2>
          </div>
          <div className="recommend-grid">
            <div className="recommend-card glass">
              <div className="rc-icon">
                <svg viewBox="0 0 24 24"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" /><path d="M19 10v2a7 7 0 0 1-14 0v-2" /><line x1="12" y1="19" x2="12" y2="23" /><line x1="8" y1="23" x2="16" y2="23" /></svg>
              </div>
              <p>歌ってみた・カラオケ音源</p>
            </div>
            <div className="recommend-card glass">
              <div className="rc-icon">
                <svg viewBox="0 0 24 24"><path d="M9 18V5l12-2v13" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="16" r="3" /></svg>
              </div>
              <p>弾いてみた・耳コピ</p>
            </div>
            <div className="recommend-card glass">
              <div className="rc-icon">
                <svg viewBox="0 0 24 24"><path d="M3 18v-6a9 9 0 0 1 18 0v6" /><path d="M21 19a2 2 0 0 1-2 2h-1a2 2 0 0 1-2-2v-3a2 2 0 0 1 2-2h3zM3 19a2 2 0 0 0 2 2h1a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2H3z" /></svg>
              </div>
              <p>楽器練習・パート別再生</p>
            </div>
          </div>
        </div>
      </section>

      {/* ===== FEATURES ===== */}
      <section className="features fade-in" id="features">
        <div className="wrap">
          <div className="features-header">
            <p className="section-label">Features</p>
            <h2 className="section-heading">特徴</h2>
          </div>
          <div className="features-grid">
            <div className="feature-card glass">
              <div className="fc-icon">
                <svg viewBox="0 0 24 24"><path d="M9 18V5l12-2v13" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="16" r="3" /></svg>
              </div>
              <h3>6パート対応</h3>
              <p>Vocals / Drums / Bass<br />Guitar / Piano / Other</p>
            </div>
            <div className="feature-card glass">
              <div className="fc-icon">
                <svg viewBox="0 0 24 24"><polyline points="16 3 21 3 21 8" /><line x1="4" y1="20" x2="21" y2="3" /><polyline points="21 16 21 21 16 21" /><line x1="15" y1="15" x2="21" y2="21" /><line x1="4" y1="4" x2="9" y2="9" /></svg>
              </div>
              <h3>抽出 &amp; 除去</h3>
              <p>取り出しも消去も<br />ワンクリックで切替</p>
            </div>
            <div className="feature-card glass">
              <div className="fc-icon">
                <svg viewBox="0 0 24 24"><rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0 1 10 0v4" /></svg>
              </div>
              <h3>オフライン処理</h3>
              <p>PC上で完結<br />サーバー送信なし</p>
            </div>
          </div>
        </div>
      </section>

      {/* ===== FAQ ===== */}
      <section className="faq fade-in" id="faq">
        <div className="wrap">
          <div className="faq-header">
            <p className="section-label">FAQ</p>
            <h2 className="section-heading">よくある質問</h2>
          </div>
          <div className="faq-list">
            {FAQ_ITEMS.map((item, idx) => (
              <div key={idx} className={`faq-item${openFaq === idx ? ' open' : ''}`}>
                <button
                  className="faq-q"
                  aria-expanded={openFaq === idx}
                  onClick={() => toggleFaq(idx)}
                >
                  {item.q}
                  <svg
                    className="faq-chevron"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M6 9l6 6 6-6" />
                  </svg>
                </button>
                <div className="faq-a">
                  <div className="faq-a-inner">{item.a}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== FOOTER ===== */}
      <footer className="footer">
        <p className="footer-copy">
          &copy; 2025 OtoWake &nbsp;|&nbsp;{' '}
          <a href="/terms.html">利用規約</a> &nbsp;|&nbsp;{' '}
          <a href="/privacy.html">プライバシーポリシー</a>
        </p>
        <ul className="footer-links">
          <li>
            <a href="https://github.com/" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
              <svg viewBox="0 0 24 24"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" /></svg>
            </a>
          </li>
          <li>
            <a href="https://twitter.com/" target="_blank" rel="noopener noreferrer" aria-label="X (Twitter)">
              <svg viewBox="0 0 24 24"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" /></svg>
            </a>
          </li>
        </ul>
      </footer>
    </>
  );
}
