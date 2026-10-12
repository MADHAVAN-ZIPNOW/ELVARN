# ELVARN search service = SearXNG (private, 127.0.0.1:8081) + proxy.py (public, $PORT, needs a token).
# The version is pinned on purpose. Never use "latest".
FROM searxng/searxng:2026.10.11-ad46ebd7d

# Production settings (the image reads them from SEARXNG_SETTINGS_PATH).
COPY --chown=977:977 settings.yml /etc/searxng/settings.yml
ENV SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml

# The gatekeeper (Python standard library only) and the script that starts both programs.
COPY --chmod=0755 proxy.py /opt/elvarn/proxy.py
COPY --chmod=0755 entrypoint.sh /opt/elvarn/entrypoint.sh

# Render sets PORT itself; 10000 is only the default for local runs.
ENV PORT=10000 \
    SEARXNG_INTERNAL_PORT=8081 \
    SEARXNG_LIMITER=false \
    SEARXNG_PUBLIC_INSTANCE=false

EXPOSE 10000

# Replace the image's own entrypoint/command with ours (ours calls the image's entrypoint for SearXNG).
ENTRYPOINT ["/bin/sh", "/opt/elvarn/entrypoint.sh"]
CMD []
