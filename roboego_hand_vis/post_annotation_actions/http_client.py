"""Keep provider HTTP diagnostics without changing request/failure semantics."""

import io
import json
import urllib.error
import urllib.parse
import urllib.request


def request_json(req, *, timeout_s, stage, model):
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(4096)
        except OSError:
            body = b''
        finally:
            exc.close()
        url = urllib.parse.urlsplit(req.full_url)
        host = url.netloc.rsplit('@', 1)[-1]
        safe_url = urllib.parse.urlunsplit((url.scheme, host, url.path, '', ''))
        authorization = req.get_header('Authorization', '')
        secrets = [authorization.removeprefix('Bearer ').strip(), url.password or '']
        secrets += [value for _, value in urllib.parse.parse_qsl(url.query)]
        secrets += [part.partition("=")[2] for part in url.query.split("&")]
        secrets += [urllib.parse.unquote(value) for value in list(secrets)]

        def clean(value):
            value = value.replace(req.full_url, safe_url)
            for secret in sorted(set(secrets), key=len, reverse=True):
                if secret:
                    value = value.replace(secret, '[REDACTED]')
            return value

        details = clean(body.decode('utf-8', errors='replace')).strip() or '(empty response body)'
        reason = f'{clean(str(exc.reason))}; stage={stage}; model={model}; url={safe_url}; response={details}'
        # Preserve HTTPError/URLError handling in existing callers.
        raise urllib.error.HTTPError(safe_url, exc.code, reason, exc.headers,
                                     io.BytesIO(details.encode('utf-8'))) from None
