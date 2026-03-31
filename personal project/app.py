from flask import Flask, render_template, request, jsonify
import flowkit as fk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import RANSACRegressor

# GUI 없는 환경을 위한 설정
matplotlib.use('Agg')

app = Flask(__name__)


# --- [Helper Functions for Logic] ---

def safe_zone_auto_gates(fsc_data, ssc_data):
    """FSC/SSC 데이터를 기반으로 Debris 제거를 위한 게이트 좌표 계산"""
    safe_mask = (fsc_data > 50000) & (fsc_data < 200000)

    if np.sum(safe_mask) < 100:
        return 0, 0, 0, 0, 50000

    safe_fsc, safe_ssc = fsc_data[safe_mask], ssc_data[safe_mask]
    bins = np.linspace(50000, 200000, 15)
    bin_idx = np.digitize(safe_fsc, bins)

    ux, uy, lx, ly = [], [], [], []

    for i in range(1, len(bins)):
        m = (bin_idx == i)
        if np.sum(m) > 100:
            x_m = safe_fsc[m].mean()
            lx.append(x_m)
            ly.append(np.percentile(safe_ssc[m], 1))

            uc = safe_ssc[m & (safe_ssc > 30000)]
            ux.append(x_m)
            uy.append(
                np.percentile(uc, 95) if len(uc) > 10
                else np.percentile(safe_ssc[m], 99)
            )

    up_c = np.polyfit(ux, uy, 1)
    low_c = np.polyfit(lx, ly, 1)

    hist, edges = np.histogram(fsc_data, bins=100, range=(0, 200000))
    d_peak = np.argmax(hist[:20])
    c_peak = d_peak + 5 + np.argmax(hist[d_peak + 5:80])
    v_idx = d_peak + np.argmin(hist[d_peak:c_peak])

    return up_c[0], up_c[1], low_c[0], low_c[1], edges[v_idx]


def generate_plot_base64(fig):
    """Matplotlib Figure를 base64 문자열로 변환"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# --- [Flask Routes] ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_info', methods=['POST'])
def upload_info():
    files = request.files.getlist('files')
    total_ev = 0

    for f in files:
        try:
            s = fk.Sample(io.BytesIO(f.read()), ignore_offset_error=True)
            total_ev += int(s.event_count)
        except:
            continue

    return jsonify({
        'count': len(files),
        'total_events': f"{total_ev:,}"
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    files = request.files.getlist('files')
    mode = request.form.get('mode')

    # --- [사용자 지정 스케일 및 기본값 처리] ---
    # 값이 비어있거나("") 숫자가 아닐 경우 기본값을 사용하도록 'or'와 try-except 활용
    try:
        fsc_limit = int(request.form.get('fsc_limit') or 250000)
        ssc_limit = int(request.form.get('ssc_limit') or 150000)
    except ValueError:
        fsc_limit, ssc_limit = 250000, 150000

    all_samples = []
    sampling_debris = []

    # 1. 데이터 로딩 및 샘플링
    for f in files:
        try:
            s = fk.Sample(io.BytesIO(f.read()), sample_id=f.filename, ignore_offset_error=True)
            all_samples.append(s)

            ev = s.get_events(source='raw')
            fi, si = s.pnn_labels.index('FSC-A'), s.pnn_labels.index('SSC-A')

            idx = np.random.choice(len(ev), min(len(ev), 5000), replace=False)
            sampling_debris.append(ev[idx][:, [fi, si]])
        except:
            continue

    if not sampling_debris:
        return jsonify({'plots': [], 'total_sum': "0"})

    total_d = np.vstack(sampling_debris)
    us, ui, ls, li, cut = safe_zone_auto_gates(total_d[:, 0], total_d[:, 1])

    plots = []
    total_cleaned_count = 0

    # 2. 분석 모드에 따른 처리
    if mode == 'debris':
        for s in all_samples:
            ev = s.get_events(source='raw')
            fi, si = s.pnn_labels.index('FSC-A'), s.pnn_labels.index('SSC-A')

            # 마스크 계산
            m_full = (ev[:, fi] > cut) & (ev[:, si] > (ls * ev[:, fi] + li)) & (ev[:, si] < (us * ev[:, fi] + ui))
            clean_count = np.sum(m_full)
            total_cleaned_count += clean_count

            # 시각화용 샘플링
            v_idx = np.random.choice(len(ev), min(len(ev), 8000), replace=False)
            fv, sv = ev[v_idx, fi], ev[v_idx, si]
            m_v = m_full[v_idx]

            # --- [시각화: 눈금 및 스케일 적용] ---
            fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor='#1a202c')
            ax.set_facecolor('black')

            ax.scatter(fv[~m_v], sv[~m_v], c='gray', s=1, alpha=0.3)
            ax.scatter(fv[m_v], sv[m_v], c='#007bff', s=1, alpha=0.5)

            # 게이트 라인 (사용자 지정 fsc_limit 반영)
            xr = np.array([0, fsc_limit])
            ax.plot(xr, us * xr + ui, 'yellow', lw=1)
            ax.plot(xr, ls * xr + li, 'cyan', lw=1)
            ax.axvline(cut, color='lime', ls='--', lw=1)

            # 축 설정
            ax.set_xlim(0, fsc_limit)
            ax.set_ylim(0, ssc_limit)

            # 눈금 및 테두리 스타일링
            ax.tick_params(axis='both', colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#4a5568')

            plots.append({
                'id': s.id,
                'img': generate_plot_base64(fig),
                'original': f"{len(ev):,}",
                'cleaned': f"{clean_count:,}"
            })

    elif mode == 'singlet':
        sampling_singlet = []
        for s in all_samples:
            ev = s.get_events(source='raw')
            fi, si = s.pnn_labels.index('FSC-A'), s.pnn_labels.index('SSC-A')
            m = (ev[:, fi] > cut) & (ev[:, si] > (ls * ev[:, fi] + li)) & (ev[:, si] < (us * ev[:, fi] + ui))
            clean_ev = ev[m]

            if len(clean_ev) > 100:
                hi, ai = s.pnn_labels.index('FSC-H'), s.pnn_labels.index('FSC-A')
                idx = np.random.choice(len(clean_ev), min(len(clean_ev), 3000), replace=False)
                sampling_singlet.append(clean_ev[idx][:, [hi, ai]])

        if not sampling_singlet:
            return jsonify({'plots': [], 'total_sum': "0"})

        total_s = np.vstack(sampling_singlet)
        ransac = RANSACRegressor(random_state=42).fit(total_s[:, 0].reshape(-1, 1), total_s[:, 1])
        err = total_s[:, 1] - ransac.predict(total_s[:, 0].reshape(-1, 1))
        low, up = np.mean(err) - (1.5 * np.std(err)), np.mean(err) + (1.5 * np.std(err))

        for s in all_samples:
            ev = s.get_events(source='raw')
            fi, si = s.pnn_labels.index('FSC-A'), s.pnn_labels.index('SSC-A')
            hi, ai = s.pnn_labels.index('FSC-H'), s.pnn_labels.index('FSC-A')

            m_d = (ev[:, fi] > cut) & (ev[:, si] > (ls * ev[:, fi] + li)) & (ev[:, si] < (us * ev[:, fi] + ui))
            clean_ev = ev[m_d]

            y_p_full = ransac.predict(clean_ev[:, hi].reshape(-1, 1))
            diff_full = clean_ev[:, ai] - y_p_full
            m_s_full = (diff_full > low) & (diff_full < up)

            singlet_count = np.sum(m_s_full)
            total_cleaned_count += singlet_count

            v_idx = np.random.choice(len(clean_ev), min(len(clean_ev), 8000), replace=False)
            vh, va = clean_ev[v_idx, hi], clean_ev[v_idx, ai]
            v_pred = ransac.predict(vh.reshape(-1, 1))
            m_s = ((va - v_pred) > low) & ((va - v_pred) < up)

            # --- [시각화: 눈금 및 스케일 적용] ---
            fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor='#1a202c')
            ax.set_facecolor('black')

            ax.scatter(vh[~m_s], va[~m_s], c='gray', s=1, alpha=0.3)
            ax.scatter(vh[m_s], va[m_s], c='#007bff', s=1, alpha=0.5)

            xr = np.linspace(0, fsc_limit, 100).reshape(-1, 1)
            yp = ransac.predict(xr)
            ax.plot(xr, yp, 'lime', lw=1)
            ax.plot(xr, yp + up, 'yellow', lw=1, ls='--')
            ax.plot(xr, yp + low, 'cyan', lw=1, ls='--')

            # 축 설정 (Singlet은 보통 가로세로 동일 스케일 사용)
            ax.set_xlim(0, fsc_limit)
            ax.set_ylim(0, ssc_limit)  # 사용자가 Singlet 탭에서 입력한 ssc_limit 적용

            ax.tick_params(axis='both', colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#4a5568')

            plots.append({
                'id': s.id,
                'img': generate_plot_base64(fig),
                'original': f"{len(clean_ev):,}",
                'cleaned': f"{singlet_count:,}"
            })

    return jsonify({'plots': plots, 'total_sum': f"{total_cleaned_count:,}"})


if __name__ == '__main__':
    app.run(port=5000, debug=True)