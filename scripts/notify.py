import os
import re
import sys
import argparse
import urllib.request
import urllib.parse
import json
from pathlib import Path

BASELINE = {
    'anomalyclip': {
        'mvtec': {
            'image_auroc': 91.5, 'image_ap': 96.2,
            'pixel_auroc': 91.1, 'pixel_aupro': 81.4
        },
        'visa': {
            'image_auroc': 82.1, 'image_ap': 85.4,
            'pixel_auroc': 95.5, 'pixel_aupro': 87.0
        },
        'mpdd': {
            'image_auroc': 77.0, 'image_ap': 82.0,
            'pixel_auroc': 96.5, 'pixel_aupro': 88.7
        },
        'btad': {
            'image_auroc': 88.3, 'image_ap': 87.3,
            'pixel_auroc': 94.2, 'pixel_aupro': 74.8
        },
        'sdd': {
            'image_auroc': 84.7, 'image_ap': 80.0,
            'pixel_auroc': 90.6, 'pixel_aupro': 67.8
        },
        'dagm': {
            'image_auroc': 97.5, 'image_ap': 92.3,
            'pixel_auroc': 95.6, 'pixel_aupro': 91.0
        },
        'dtd-synthetic': {
            'image_auroc': 93.5, 'image_ap': 97.0,
            'pixel_auroc': 97.9, 'pixel_aupro': 92.3
        },
        'dagm_kaggleupload': {
            'image_auroc': 97.5, 'image_ap': 92.3,
            'pixel_auroc': 95.6, 'pixel_aupro': 91.0
        }
    }
}

METRIC_KEYS = ['image_auroc', 'image_ap', 'pixel_auroc', 'pixel_aupro']


def parse_log(log_path):
    """Extract mean metrics from the last table in a log file."""
    text = Path(log_path).read_text()
    # Look for the pipe-formatted results table; grab the 'mean' row
    # Pattern: | mean | 94.3 | 88.5 | 78.3 | 81.9 |
    pattern = r'\|\s*mean\s*\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|'
    matches = re.findall(pattern, text)
    if not matches:
        # try 2-column (image-level only)
        pattern2 = r'\|\s*mean\s*\|([^|]+)\|([^|]+)\|'
        matches2 = re.findall(pattern2, text)
        if matches2:
            vals = [v.strip() for v in matches2[-1]]
            return {'image_auroc': float(vals[0]), 'image_ap': float(vals[1])}
        return {}

    vals = [v.strip() for v in matches[-1]]
    keys = ['pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap']
    result = {}
    for k, v in zip(keys, vals):
        try:
            result[k] = float(v)
        except ValueError:
            pass
    return result


def fmt_delta(val, base):
    if base is None or val is None:
        return ''
    d = val - base
    sign = '+' if d >= 0 else ''
    return f'({sign}{d:.1f})'


def send_telegram(token, chat_id, text):
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    data = urllib.parse.urlencode({
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'HTML',
    }).encode()
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            if not result.get('ok'):
                print(f'Telegram error: {result}', file=sys.stderr)
    except Exception as e:
        print(f'Telegram send failed: {e}', file=sys.stderr)


def load_env():
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='test', choices=['test', 'msg'],
                        help='"test": parse log and compare vs baseline; "msg": send raw message')
    # test mode
    parser.add_argument('--dataset',  default='', help='dataset name (visa/mpdd/dtd/...)')
    parser.add_argument('--log_path', default='', help='path to test log.txt')
    parser.add_argument('--exp_name', default='', help='experiment label')
    # msg mode
    parser.add_argument('--text', default='', help='raw message text (used with --mode msg)')
    args = parser.parse_args()

    load_env()
    token   = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print('TELEGRAM_TOKEN / TELEGRAM_CHAT_ID not set', file=sys.stderr)
        sys.exit(1)

    if args.mode == 'msg':
        send_telegram(token, chat_id, args.text or '(empty message)')
        return

    # --- test mode (original behavior) ---
    if not args.dataset or not args.log_path:
        print('--dataset and --log_path required in test mode', file=sys.stderr)
        sys.exit(1)

    metrics = parse_log(args.log_path)
    if not metrics:
        print(f'Could not parse metrics from {args.log_path}', file=sys.stderr)
        sys.exit(1)

    ds_key = args.dataset.lower()
    baseline = BASELINE.get('anomalyclip', {}).get(ds_key, {})

    label = args.exp_name or args.dataset
    lines = [f'<b>✅ {label} → {args.dataset.upper()}</b>']
    lines.append('')
    lines.append(f'{"Metric":<14} {"Ours":>7} {"Base":>7} {"Δ":>7}')
    lines.append('─' * 38)

    for k in METRIC_KEYS:
        v    = metrics.get(k)
        base = baseline.get(k)
        if v is None:
            continue
        base_str  = f'{base:.1f}' if base is not None else '  -  '
        delta_str = fmt_delta(v, base)
        lines.append(f'{k:<14} {v:>7.1f} {base_str:>7} {delta_str:>7}')

    msg = '\n'.join(lines)
    print(msg)
    send_telegram(token, chat_id, msg)


if __name__ == '__main__':
    main()
