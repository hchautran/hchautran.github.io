import { pathToRoot } from "../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const Navbar: QuartzComponent = ({ fileData }: QuartzComponentProps) => {
  const baseDir = pathToRoot(fileData.slug!)
  return (
    <nav class="navbar">
      <a class="navbar-home" href={baseDir}></a>
      <div class="navbar-links">
        <a href={baseDir}>About</a>
        <a href={`${baseDir}/notes/`}>Notes</a>
        <a href={`${baseDir}#-news`}>News</a>
        <a href={`${baseDir}#-publications`}>Publications</a>
      </div>
    </nav>
  )
}


Navbar.css = `
body {
  padding-top: 56px;
}

.navbar {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.7rem 1.5rem;
  background: var(--light);
  border-bottom: 1px solid var(--lightgray);
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
  box-sizing: border-box;
  transition: box-shadow 0.25s ease, background 0.25s ease;
}

.navbar:hover {
  box-shadow: 0 4px 18px rgba(27, 33, 48, 0.06);
}

.navbar-home {
  font-weight: 700;
  font-size: 1rem;
  color: var(--darkgray);
  text-decoration: none;
  white-space: nowrap;
  letter-spacing: 0.01em;
  transition: color 0.2s ease;
}

.navbar-home:hover {
  color: var(--secondary);
}

.navbar-links {
  display: flex;
  align-items: center;
  gap: 1.6rem;
}

.navbar-links a {
  position: relative;
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--darkgray);
  text-decoration: none;
  letter-spacing: 0.01em;
  padding: 0.25rem 0;
  transition: color 0.2s ease;
}

.navbar-links a::after {
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 1.5px;
  background: var(--secondary);
  transform: scaleX(0);
  transform-origin: center;
  transition: transform 0.25s ease;
}

.navbar-links a:hover {
  color: var(--secondary);
}

.navbar-links a:hover::after {
  transform: scaleX(1);
}

@media (max-width: 480px) {
  .navbar {
    padding: 0.6rem 1rem;
  }
  .navbar-links {
    gap: 1rem;
  }
  .navbar-links a {
    font-size: 0.82rem;
  }
}
`

export default (() => Navbar) satisfies QuartzComponentConstructor
