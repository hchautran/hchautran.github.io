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
  padding-top: 48px;
}

.navbar {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.6rem 1.2rem;
  border-bottom: 1px solid var(--lightgray);
  background: var(--light);
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
  box-sizing: border-box;
}

.navbar-home {
  font-weight: 700;
  font-size: 1rem;
  color: var(--darkgray);
  text-decoration: none;
  white-space: nowrap;
}

.navbar-home:hover {
  color: var(--secondary);
}

.navbar-links {
  display: flex;
  align-items: center;
  gap: 1.2rem;
}

.navbar-links a {
  font-size: 0.9rem;
  color: var(--gray);
  text-decoration: none;
  transition: color 0.2s;
}

.navbar-links a:hover {
  color: var(--secondary);
}

@media (max-width: 480px) {
  .navbar-links {
    gap: 0.7rem;
  }
  .navbar-links a {
    font-size: 0.8rem;
  }
}
`

export default (() => Navbar) satisfies QuartzComponentConstructor
