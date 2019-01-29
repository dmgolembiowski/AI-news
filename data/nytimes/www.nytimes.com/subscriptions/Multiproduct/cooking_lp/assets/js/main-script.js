// JavaScript Document
var params = unescape(window.location.search.replace('?', '&').replace(/OC=\d+&/, ''));

(function ($) { 

    $(document).ready(function(){

        var cookingVideo = document.getElementById("cookingVideo");
        
            document.getElementById('cookingVideo').pause();
        
        /* ############### For loading video on modal pop up ########### */  
        
            // Get the modal
            var modal = document.getElementById('CookingVideoModal');

            // Get the button that opens the modal
            var btn = document.getElementById("cookingGuide");

            // Get the <span> element that closes the modal
            var span = document.getElementsByClassName("close")[0];

            // When the user clicks on the button, open the modal 
            btn.onclick = function() {
                modal.style.display = "block";
                //this.played?this.pause():this.play();
                //document.getElementById('cookingVideo').play();
                //$("#cookingVideo").play();
                //$("#cookingVideo").attr({autoplay:"autoplay", playsinline: "playsinline"});
                //cookingVideo.play();
                //document.getElementsByTagName('video')[0].play();
                var videoHeight =$("#cookingVideo").innerHeight();
                console.log('video height:'+videoHeight);
            }

            // When the user clicks on <span> (x), close the modal
            span.onclick = function() {           
                modal.style.display = "none";            
                
            }

            // When the user clicks anywhere outside of the modal, close it
            window.onclick = function(event) {
                //document.getElementById('cookingVideo').pause();
                if (event.target == modal) {              
                    modal.style.display = "none";
                    cookingVideo.pause();
                }
            }              
        
            $('#cookingVideo').click(function(){
                //this.paused?this.play():this.pause();
            });
        
            $('.close').click(function(){
                cookingVideo.pause();
            });
        
            
        
        // End of Modal Video /////////////////////////////////////////////
        
            
        /// NAVIGATION //// 
        function makeLinks() {
            /*

            */
            link_cta = 'https://www.nytimes.com/subscription/multiproduct/lp8HYKU.html?campaignId=6JUHR'; 
            link_why = 'lp8FW9H.html?campaignId='+cid; 
            link_what = 'lp3L3W6.html?campaignId='+cid; 
            //certain links .. might need to dynamically add cid.. prob better to hard code it. 
            link_newreader = 'https://www.nytimes.com/subscription/multiproduct/lp8HYKU.html?campaignId=6JUHR'; 
            link_education = 'https://www.nytimes.com/subscriptions/edu/lp8LQFK.html?campaignId=6RR8R'; 
            link_spotify = 'https://www.nytimes.com/subscriptions/Multiproduct/lp8U939.html?campaignId=6RQWY'; 
            link_corporate = 'http://nytimesgroupsubscriptions.com/?Pardot_Campaign_Code_Form_Input=74LHR'; 
            link_crosswords = 'https://www.nytimes.com/subscriptions/games/lp897H9.html?campaignId=6RRWR'; 
            link_gift = 'https://www.nytimes.com/subscriptions/Multiproduct/lp8R34X.html?campaignId=3FQ4Y';
            $('.link_education').attr('href',link_education); 
            $('.link_newreader').attr('href',link_newreader); 
            $('.link_spotify, #listenin .discover').attr('href',link_spotify); 
            $('.link_crosswords').attr('href',link_crosswords); 
            $('.link_corporate').attr('href',link_corporate); 
            $('.link_gift').attr('href',link_gift);

            $('.link_cta,.sub_cta').attr('href',link_cta); 
            $('.link_why').attr('href',link_why); 
            $('.link_what').attr('href',link_what);

                ///so.. some links open in new window.
            $('footer a,a.external').attr('target','_blank'); 

        }

            
        //footer 

        function enableFooterAccord(){
            $("footer.footer .col h3 button.footer_toggle").on("click", function(){
                var theseLinks = $(this).parents(".col").find(".footer_links");
                var thisBtn =$(this).parents("h3");
                var thisIcon = $(this).parents("h3").find("i");
                var thisSvg = $(this).find("i svg g");

                if(thisBtn.hasClass("active")){
                    thisBtn.removeClass("active");
                    theseLinks.slideUp();
                    thisIcon.transition({rotate:"0"});
                }else{
                    thisBtn.addClass("active");
                    theseLinks.slideDown();
                    thisIcon.transition({rotate:"45"});
                }

            });//END of mobile_nav_links h5 click

        }//END of enableMenu

        function makeFooter() {
        var placePut = $('footer'); 
        var url_nav = 'cooking_lp/html/footer.html'; 
        $.get(url_nav, function(connie) {

                placePut.html(connie); 
                enableFooterAccord();
                makeLinks(); 
                //assure the Links will work.. call to setLinks
        }); 

        }
        //end of  footer 
        
        function enableFaq(){
            $('.question').on("click", function () {
                $(this).toggleClass('expanded').next('.answer').slideToggle();
            });
        }//End of enableFaq

        function getParameterByName(name)
        {
            name = name.replace(/[\[]/, "\\\[").replace(/[\]]/, "\\\]");
            var regexS = "[\\?&]" + name + "=([^&#]*)";
            var regex = new RegExp(regexS);
            var results = regex.exec(window.location.search);
            if(results == null)
                return "";
            else
                return decodeURIComponent(results[1].replace(/\+/g, " "));
        }
        
        $("a.down_arrow").on("click", function(){  
            var nextSection  = $(this).parents("section").next("section").attr("id");
            
            $('html, body').animate({
                scrollTop: $("section#"+nextSection).offset().top - 0
            }, 800);       
        });
        
        $(".subscribe-now,.subscribe-now-mobile").on("click", function(){  
            var nextSection  = 'subscribeNYT';
            
            $('html, body').animate({
                scrollTop: $("section#"+nextSection).offset().top -0
            }, 800);       
        });
    // End of down arrow 
        
        makeFooter();
        enableFaq();
        addLinkParameters('.button--purchase');
        //checkForAP();    
            
    }); 

})(jQuery);