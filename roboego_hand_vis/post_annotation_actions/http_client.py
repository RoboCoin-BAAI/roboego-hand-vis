"""Request progress and safe provider diagnostics, preserving exception types."""

import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def _safe_context(req):
    url = urllib.parse.urlsplit(req.full_url)
    host = url.netloc.rsplit('@', 1)[-1]
    safe_url = urllib.parse.urlunsplit((url.scheme, host, url.path, '', ''))
    authorization = req.get_header('Authorization', '')
    secrets = [authorization.removeprefix('Bearer ').strip(), url.password or '']
    secrets += [value for _, value in urllib.parse.parse_qsl(url.query)]
    secrets += [part.partition('=')[2] for part in url.query.split('&')]
    secrets += [urllib.parse.unquote(value) for value in list(secrets)]

    def clean(value):
        value = str(value).replace(req.full_url, safe_url)
        for secret in sorted(set(secrets), key=len, reverse=True):
            if secret:
                value = value.replace(secret, '[REDACTED]')
        return value
    return safe_url, clean


def request_json(req, *, timeout_s, stage, model):
    safe_url, clean = _safe_context(req)
    started = time.monotonic()
    request_bytes = len(req.data or b'')
    image_count = 0
    try:
        body = json.loads(req.data or b'{}')
        for message in body.get('messages', []):
            content = message.get('content', [])
            if isinstance(content, list):
                image_count += sum(isinstance(item, dict) and item.get('type') == 'image_url' for item in content)
    except (ValueError, TypeError, AttributeError):
        pass
    phase = 'urlopen'

    def progress(event, **extra):
        print(json.dumps({
            'event': event, 'stage': clean(stage), 'model': clean(model), 'url': safe_url,
            'request_bytes': request_bytes, 'image_count': image_count, 'timeout_s': timeout_s,
            'elapsed_s': round(time.monotonic() - started, 3), 'phase': phase, **extra,
        }, ensure_ascii=False), file=sys.stderr, flush=True)

    progress('request_start')
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            phase = 'response_read'
            payload = json.loads(response.read().decode('utf-8'))
        progress('request_complete')
        return payload
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(4096)
        except OSError:
            body = b''
        finally:
            exc.close()
        details = clean(body.decode('utf-8', errors='replace')).strip() or '(empty response body)'
        reason = f'{clean(exc.reason)}; stage={clean(stage)}; model={clean(model)}; url={safe_url}; response={details}'
        progress('request_http_error', http_status=exc.code)
        raise urllib.error.HTTPError(safe_url, exc.code, reason, exc.headers,
                                     io.BytesIO(details.encode('utf-8'))) from None
    except TimeoutError as exc:
        elapsed = time.monotonic() - started
        progress('request_timeout')
        raise TimeoutError(
            f'{clean(exc)}; stage={clean(stage)}; model={clean(model)}; '
            f'timeout_s={timeout_s}; elapsed_s={elapsed:.3f}; phase={phase}; url={safe_url}'
        ) from None
    except urllib.error.URLError as exc:
        progress('request_network_error')
        raise urllib.error.URLError(
            f'{clean(exc.reason)}; stage={clean(stage)}; model={clean(model)}; '
            f'timeout_s={timeout_s}; phase={phase}; url={safe_url}'
        ) from None
