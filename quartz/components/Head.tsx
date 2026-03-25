import { i18n } from "../i18n"
import { FullSlug, joinSegments, pathToRoot } from "../util/path"
import { JSResourceToScriptElement } from "../util/resources"
import { googleFontHref } from "../util/theme"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

export default (() => {
  const Head: QuartzComponent = ({ cfg, fileData, externalResources }: QuartzComponentProps) => {
    const title = fileData.frontmatter?.title ?? i18n(cfg.locale).propertyDefaults.title
    const description =
      fileData.description?.trim() ?? i18n(cfg.locale).propertyDefaults.description
    const { css, js } = externalResources

    const url = new URL(`https://${cfg.baseUrl ?? "example.com"}`)
    const path = url.pathname as FullSlug
    const baseDir = fileData.slug === "404" ? path : pathToRoot(fileData.slug!)

    const iconPath = joinSegments(baseDir, "static/icon.svg")
    const ogImagePath = `https://${cfg.baseUrl}/static/og-image.png`

    return (
      <head>
        <title>{title}</title>
        <meta charSet="utf-8" />
        <style dangerouslySetInnerHTML={{__html: `
          #loading-skeleton {
            position: fixed;
            inset: 0;
            z-index: 9999;
            background: #faf8f8;
            display: flex;
            flex-direction: row;
            gap: 2rem;
            padding: 5rem 2rem 2rem;
            overflow: hidden;
            transition: opacity 0.25s ease;
          }
          html[saved-theme="dark"] #loading-skeleton {
            background: #1c1c1e;
          }
          #loading-skeleton.sk-hidden {
            opacity: 0;
            pointer-events: none;
          }
          .sk-bar {
            border-radius: 5px;
            background: linear-gradient(90deg, #e8e8e8 25%, #f4f4f4 50%, #e8e8e8 75%);
            background-size: 200% 100%;
            animation: sk-shimmer 1.4s infinite linear;
          }
          html[saved-theme="dark"] .sk-bar {
            background: linear-gradient(90deg, #2c2c2e 25%, #3a3a3c 50%, #2c2c2e 75%);
            background-size: 200% 100%;
          }
          @keyframes sk-shimmer {
            0%   { background-position: 200% 0; }
            100% { background-position: -200% 0; }
          }
          .sk-sidebar {
            width: 380px;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
            padding-top: 0.5rem;
          }
          .sk-center {
            flex: 1;
            min-width: 0;
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
          }
          @media (max-width: 1510px) { .sk-sidebar.sk-right { display: none; } }
          @media (max-width: 1000px) { .sk-sidebar { display: none; } }
        `}} />
        <script dangerouslySetInnerHTML={{__html: `
          (function () {
            function hideSkeleton() {
              var sk = document.getElementById('loading-skeleton');
              if (!sk) return;
              sk.classList.add('sk-hidden');
              setTimeout(function () { sk.style.display = 'none'; }, 280);
            }
            function showSkeleton() {
              var sk = document.getElementById('loading-skeleton');
              if (!sk) return;
              sk.style.display = 'flex';
              requestAnimationFrame(function () { sk.classList.remove('sk-hidden'); });
            }
            window.addEventListener('load', hideSkeleton);
            document.addEventListener('nav', hideSkeleton);
            document.addEventListener('click', function (e) {
              var a = e.target && e.target.closest && e.target.closest('a');
              if (!a || a.getAttribute('target') === '_blank') return;
              if (a.dataset && 'routerIgnore' in a.dataset) return;
              try {
                var url = new URL(a.href);
                if (url.origin === location.origin && url.pathname !== location.pathname) {
                  showSkeleton();
                }
              } catch (_) {}
            });
          })();
        `}} />

        {cfg.theme.cdnCaching && cfg.theme.fontOrigin === "googleFonts" && (
          <>
            <link rel="preconnect" href="https://fonts.googleapis.com" />
            <link rel="preconnect" href="https://fonts.gstatic.com" />
            <link rel="stylesheet" href={googleFontHref(cfg.theme)} />
          </>
        )}
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta property="og:title" content={title} />
        <meta property="og:description" content={description} />
        {cfg.baseUrl && <meta property="og:image" content={ogImagePath} />}
        <meta property="og:width" content="1200" />
        <meta property="og:height" content="675" />
        <link rel="icon" href={iconPath} />
        <meta name="description" content={description} />
        <meta name="generator" content="Quartz" />
        <link rel="stylesheet" type="text/css" href="https://tikzjax.com/v1/fonts.css" />
        <script src="https://tikzjax.com/v1/tikzjax.js"></script>
        <style>{`.tikz-center svg { display: block; margin: 0 auto; }`}</style>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" spa-preserve={true}></script>
        <script
          spa-preserve={true}
          dangerouslySetInnerHTML={{
            __html: `document.addEventListener('nav',function(){document.querySelectorAll('script[data-plot-script]').forEach(function(el){try{(new Function(el.textContent))()}catch(e){console.error('Plot init error:',e)}})});`,
          }}
        ></script>
        {css.map((href) => (
          <link key={href} href={href} rel="stylesheet" type="text/css" spa-preserve />
        ))}
        {js
          .filter((resource) => resource.loadTime === "beforeDOMReady")
          .map((res) => JSResourceToScriptElement(res, true))}
      </head>
    )
  }

  return Head
}) satisfies QuartzComponentConstructor
