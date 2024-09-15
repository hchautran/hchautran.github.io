import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import style from "./styles/backlinks.scss"

const Backlinks: QuartzComponent = ({
  fileData,
  allFiles,
  displayClass,
  cfg,
}: QuartzComponentProps) => {
  return (
      <script type="text/javascript" id="clustrmaps" src="//clustrmaps.com/map_v2.js?d=VV_9AxgUp6rEn_vluenH0AQsjlWKPLdDalJIaJg96ms&cl=ffffff&w=a"></script>
  )
}

Backlinks.css = style
export default (() => Backlinks) satisfies QuartzComponentConstructor
